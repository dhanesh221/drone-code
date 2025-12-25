import cv2
from pymavlink import mavutil
from ultralytics import YOLO
import threading
import time

# === CONFIG ===
connection_string = '/dev/ttyACM0'  # Change if needed
RTSP_STREAM_URL = "rtsp://192.168.144.25:8554/main.264"
MODEL_PATH = "/home/sherlock/best.pt"
CONFIDENCE_THRESHOLD = 0.5
# ==============

# Load YOLOv8 model
model = YOLO(MODEL_PATH)
class_names = model.names

# Shared telemetry data
telemetry_data = {'lat': None, 'lon': None, 'alt': None}
telemetry_lock = threading.Lock()
stop_flag = False


def telemetry_thread():
    global telemetry_data, stop_flag
    print("[DEBUG] Connecting to MAVLink on:", connection_string)
    master = mavutil.mavlink_connection(connection_string)
    print("[DEBUG] Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"[INFO] Heartbeat from system {master.target_system}, component {master.target_component}")

    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        1,  # Hz
        1   # Start stream
    )

    while not stop_flag:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
        if msg:
            with telemetry_lock:
                telemetry_data['lat'] = msg.lat / 1e7
                telemetry_data['lon'] = msg.lon / 1e7
                telemetry_data['alt'] = msg.relative_alt / 1000  # in meters
        time.sleep(0.1)


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

    print("[INFO] Video stream started with object detection and telemetry overlay.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame from RTSP.")
                break

            # === Object Detection ===
            results = model(frame)[0]
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # === Telemetry Overlay ===
            with telemetry_lock:
                lat = telemetry_data['lat']
                lon = telemetry_data['lon']
                alt = telemetry_data['alt']

            if lat is not None and lon is not None and alt is not None:
                gps_text = f"Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt:.1f} m"
            else:
                gps_text = "Waiting for GPS..."

            cv2.putText(frame, gps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # === Display Frame ===
            cv2.imshow("YOLOv8 + Telemetry (RTSP)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        stop_flag = True
        thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Clean exit.")


if __name__ == "__main__":
    main()
