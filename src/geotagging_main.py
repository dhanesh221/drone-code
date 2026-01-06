import cv2
from pymavlink import mavutil
from ultralytics import YOLO
import threading
import time
import math
import os
import numpy as np
from datetime import datetime
import folium
from folium.plugins import MarkerCluster

# === CONFIG ===
connection_string = '/dev/ttyACM0'
RTSP_STREAM_URL = "rtsp://192.168.144.25:8554/main.264"
MODEL_PATH = "/home/sherlock/best.pt"
CONFIDENCE_THRESHOLD = 0.5
INFERENCE_SKIP = 1
CAMERA_HORIZONTAL_FOV_DEG = 80
FPS = 30
# ==============

model = YOLO(MODEL_PATH)
class_names = model.names

telemetry_data = {
    'lat': None,
    'lon': None,
    'alt': None,
    'pitch': None,
    'battery_voltage': None,
    'battery_percent': None
}

telemetry_lock = threading.Lock()
stop_flag = False
detected_objects = []

def telemetry_thread():
    global telemetry_data, stop_flag
    print("[DEBUG] Connecting to MAVLink on:", connection_string)
    master = mavutil.mavlink_connection(connection_string)
    master.wait_heartbeat()
    print(f"[INFO] Heartbeat from system {master.target_system}, component {master.target_component}")

    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        10,
        1
    )

    while not stop_flag:
        try:
            msg = master.recv_match(blocking=True, timeout=1)
            if not msg:
                continue

            with telemetry_lock:
                if msg.get_type() == "GLOBAL_POSITION_INT":
                    telemetry_data['lat'] = msg.lat / 1e7
                    telemetry_data['lon'] = msg.lon / 1e7
                    telemetry_data['alt'] = msg.relative_alt / 1000.0
                elif msg.get_type() == "ATTITUDE":
                    telemetry_data['pitch'] = math.degrees(msg.pitch)
                elif msg.get_type() == "SYS_STATUS":
                    telemetry_data['battery_voltage'] = msg.voltage_battery / 1000.0
                    telemetry_data['battery_percent'] = msg.battery_remaining
        except Exception as e:
            print("[ERROR] Telemetry thread:", e)
        time.sleep(0.05)

def pixel_to_gps(x_px, y_px, frame_w, frame_h, lat, lon, alt_m, pitch_deg, fov_deg=80):
    pitch = math.radians(pitch_deg)
    fov_rad = math.radians(fov_deg)
    aspect_ratio = frame_h / frame_w
    ground_width = 2 * alt_m * math.tan(fov_rad / 2)
    ground_height = ground_width * aspect_ratio
    mpp_x = ground_width / frame_w
    mpp_y = ground_height / frame_h

    dx = (x_px - frame_w / 2) * mpp_x
    dy = (y_px - frame_h / 2) * mpp_y
    dz = -alt_m

    x1 = dx
    y1 = dy
    z1 = dz

    x2 = x1 * math.cos(pitch) + z1 * math.sin(pitch)
    y2 = y1
    z2 = -x1 * math.sin(pitch) + z1 * math.cos(pitch)

    scale = -alt_m / z2 if z2 != 0 else 0
    x_ground = x2 * scale
    y_ground = y2 * scale

    delta_lat = y_ground / 111320
    delta_lon = x_ground / (40075000 * math.cos(math.radians(lat)) / 360)
    obj_lat = lat + delta_lat
    obj_lon = lon + delta_lon

    return obj_lat, obj_lon

def main():
    global stop_flag
    thread = threading.Thread(target=telemetry_thread)
    thread.start()

    cap = cv2.VideoCapture(RTSP_STREAM_URL)
    if not cap.isOpened():
        print(f"[ERROR] Could not open RTSP stream at: {RTSP_STREAM_URL}")
        stop_flag = True
        thread.join()
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame from RTSP.")
        stop_flag = True
        thread.join()
        return

    FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    output_path = os.path.join(downloads_folder, f"drone_output_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    print(f"[INFO] Saving output video to: {output_path}")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame. Attempting reconnection...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_STREAM_URL)
                retry_count = 0
                while not cap.isOpened() and retry_count < 5:
                    print(f"[INFO] Retry {retry_count + 1}")
                    time.sleep(2)
                    cap = cv2.VideoCapture(RTSP_STREAM_URL)
                    retry_count += 1
                if not cap.isOpened():
                    black_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    out.write(black_frame)
                    continue
                else:
                    print("[INFO] Reconnected.")
                    continue

            if frame_count % INFERENCE_SKIP == 0:
                results = model(frame)[0]

                with telemetry_lock:
                    lat = telemetry_data.get('lat')
                    lon = telemetry_data.get('lon')
                    alt = telemetry_data.get('alt')
                    pitch = telemetry_data.get('pitch')

                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < CONFIDENCE_THRESHOLD:
                        continue

                    cls_id = int(box.cls[0])
                    class_name = class_names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


                    if all(v is not None for v in [lat, lon, alt, pitch]):
                        obj_lat, obj_lon = pixel_to_gps(
                            cx, cy, FRAME_WIDTH, FRAME_HEIGHT,
                            lat, lon, alt, pitch,
                            fov_deg=CAMERA_HORIZONTAL_FOV_DEG
                        )
                        gps_str = f"{class_name} at {obj_lat:.6f}, {obj_lon:.6f}"
                        print("[OBJECT DETECTED]", gps_str)

                        detected_objects.append({
                            'class': class_name,
                            'confidence': conf,
                            'latitude': obj_lat,
                            'longitude': obj_lon,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })

                        cv2.putText(frame, f"GPS: {obj_lat:.6f}, {obj_lon:.6f}",
                                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            with telemetry_lock:
                lat = telemetry_data.get('lat')
                lon = telemetry_data.get('lon')
                alt = telemetry_data.get('alt')
                voltage = telemetry_data.get('battery_voltage')
                percent = telemetry_data.get('battery_percent')

            gps_text = f"Drone: Lat {lat:.6f}, Lon {lon:.6f}, Alt {alt:.1f} m" if lat and lon and alt else "Waiting for GPS..."
            battery_text = f"Battery: {voltage:.1f}V ({percent}%)" if voltage and percent is not None else "Battery: N/A"

            cv2.putText(frame, gps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, battery_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("YOLOv8 + Telemetry (RTSP)", frame)
            if frame.shape[0] == FRAME_HEIGHT and frame.shape[1] == FRAME_WIDTH:
                out.write(frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        stop_flag = True
        thread.join()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("[INFO] Video saved to:", output_path)

        if detected_objects:
            first_obj = detected_objects[0]
            fmap = folium.Map(location=[first_obj['latitude'], first_obj['longitude']], zoom_start=17)
            marker_cluster = MarkerCluster().add_to(fmap)

            for obj in detected_objects:
                folium.Marker(
                    location=[obj['latitude'], obj['longitude']],
                    popup=f"{obj['class']} ({obj['confidence']:.2f}) at {obj['timestamp']}",
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(marker_cluster)

            map_path = os.path.join(downloads_folder, f"drone_map_{timestamp}.html")
            fmap.save(map_path)
            print("[INFO] Detection map saved to:", map_path)
        else:
            print("[INFO] No detections — map not created.")

        print("[INFO] Clean exit.")

if __name__ == "__main__":
    main()
