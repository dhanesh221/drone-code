import cv2
from ultralytics import YOLO
from dronekit import connect
import threading
import time

# === USER SETTINGS ===
VIDEO_STREAM_URL = "rtsp://192.168.144.25:8554/main.264"
TELEMETRY_CONNECTION = "tcp:127.0.0.1:5760"

OUTPUT_VIDEO = "output_detected_video.mp4"
FRAME_RATE = 30
CONF_THRESHOLD = 0.5
TELEMETRY_INTERVAL = 1  # Seconds
# =====================

# Load YOLOv8 model
model = YOLO("/home/sherlock/best.pt")
class_names = model.names

# Connect to Pixhawk
print("[INFO] Connecting to Pixhawk telemetry...")
try:
    vehicle = connect(TELEMETRY_CONNECTION, wait_ready=True, timeout=60)
    print("[INFO] Connected to Pixhawk.")
except Exception as e:
    print(f"[ERROR] Could not connect to telemetry: {e}")
    exit(1)

# Shared telemetry data container
telemetry_data = {
    "lat": "N/A",
    "lon": "N/A",
    "alt": "N/A",
    "voltage": "N/A",
    "current": "N/A",
    "level": "N/A"
}
telemetry_lock = threading.Lock()
stop_flag = False

# Thread to continuously update telemetry data
def telemetry_updater():
    global telemetry_data, stop_flag
    while not stop_flag:
        try:
            gps = vehicle.location.global_frame
            battery = vehicle.battery

            with telemetry_lock:
                telemetry_data["lat"] = f"{gps.lat:.6f}" if gps.lat else "N/A"
                telemetry_data["lon"] = f"{gps.lon:.6f}" if gps.lon else "N/A"
                telemetry_data["alt"] = f"{gps.alt:.1f}" if gps.alt else "N/A"
                telemetry_data["voltage"] = f"{battery.voltage:.1f}" if battery.voltage else "N/A"
                telemetry_data["current"] = f"{battery.current:.1f}" if battery.current else "N/A"
                telemetry_data["level"] = f"{battery.level:.0f}" if battery.level else "N/A"

            print(f"[TELEMETRY] GPS: ({telemetry_data['lat']}, {telemetry_data['lon']}, {telemetry_data['alt']}m) "
                  f"| Battery: {telemetry_data['voltage']}V, {telemetry_data['current']}A, {telemetry_data['level']}%")
        except Exception as e:
            print(f"[WARNING] Telemetry read failed: {e}")
        time.sleep(TELEMETRY_INTERVAL)

# Start telemetry thread
telemetry_thread = threading.Thread(target=telemetry_updater)
telemetry_thread.start()

# Start video stream
cap = cv2.VideoCapture(VIDEO_STREAM_URL)
if not cap.isOpened():
    print("[ERROR] Could not open video stream.")
    stop_flag = True
    vehicle.close()
    exit(1)

# Prepare video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
frame_size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FRAME_RATE, frame_size)

print("[INFO] Starting video + detection + telemetry overlay...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame read failed.")
            break

        # Run detection
        results = model(frame)
        boxes = results[0].boxes
        detected = False

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            cls = int(box.cls[0])
            class_name = class_names[cls]

            if confidence > CONF_THRESHOLD:
                detected = True
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Overlay telemetry ---
        with telemetry_lock:
            telemetry_text = f"GPS: {telemetry_data['lat']}, {telemetry_data['lon']}, Alt: {telemetry_data['alt']} m"
            battery_text = f"Battery: {telemetry_data['voltage']} V, {telemetry_data['current']} A, {telemetry_data['level']}%"

        cv2.putText(frame, telemetry_text, (10, frame_size[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, battery_text, (10, frame_size[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if detected:
            out.write(frame)

        cv2.imshow("YOLOv8 + Telemetry", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

finally:
    stop_flag = True
    telemetry_thread.join()
    vehicle.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[INFO] Clean exit. Video saved to:", OUTPUT_VIDEO)
