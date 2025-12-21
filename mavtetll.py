import cv2
from ultralytics import YOLO
import asyncio
import threading
import time
from mavsdk import System

# === USER SETTINGS ===
VIDEO_STREAM_URL = "rtsp://192.168.144.25:8554/main.264"
OUTPUT_VIDEO = "output_detected_video.mp4"
FRAME_RATE = 30
CONF_THRESHOLD = 0.5
# =====================

# Load YOLOv8 model
model = YOLO("/home/sherlock/yolov8m.pt")
class_names = model.names

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

# Flag to stop threads/tasks
stop_flag = False

# MAVSDK async telemetry updater
async def telemetry_updater(drone):
    global telemetry_data, stop_flag

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[INFO] Global position and home position are OK")
            break

    battery_task = asyncio.create_task(drone.telemetry.battery())
    position_task = asyncio.create_task(drone.telemetry.position())

    while not stop_flag:
        try:
            position = await position_task.__anext__()
            battery = await battery_task.__anext__()

            with telemetry_lock:
                telemetry_data["lat"] = f"{position.latitude_deg:.6f}"
                telemetry_data["lon"] = f"{position.longitude_deg:.6f}"
                telemetry_data["alt"] = f"{position.relative_altitude_m:.1f}"
                telemetry_data["voltage"] = f"{battery.voltage_v:.1f}"
                telemetry_data["current"] = f"{battery.current_a:.1f}" if battery.current_a is not None else "N/A"
                telemetry_data["level"] = f"{battery.remaining_percent*100:.0f}" if battery.remaining_percent is not None else "N/A"

            print(f"[TELEMETRY] GPS: ({telemetry_data['lat']}, {telemetry_data['lon']}, {telemetry_data['alt']}m) "
                  f"| Battery: {telemetry_data['voltage']}V, {telemetry_data['current']}A, {telemetry_data['level']}%")

        except Exception as e:
            print(f"[WARNING] Telemetry read failed: {e}")
            await asyncio.sleep(1)  # avoid busy loop on errors

# Function to run async loop in separate thread
def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Main function
def main():
    global stop_flag

    # Create MAVSDK System object
    drone = System()

    # Start asyncio loop in separate thread
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_async_loop, args=(loop,), daemon=True)
    t.start()

    # Connect to drone on MAVLink UDP
    async def connect_drone():
        print("[INFO] Connecting to drone via MAVSDK...")
        await drone.connect(system_address="tcp:127.0.0.1:5760")

        print("[INFO] Drone connected.")

    # Run connect coroutine in loop
    asyncio.run_coroutine_threadsafe(connect_drone(), loop).result()

    # Start telemetry updater task
    telemetry_task = asyncio.run_coroutine_threadsafe(telemetry_updater(drone), loop)

    # Open video stream
    cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    if not cap.isOpened():
        print("[ERROR] Could not open video stream.")
        stop_flag = True
        loop.call_soon_threadsafe(loop.stop)
        t.join()
        return

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

            # Write every frame for smooth output
            out.write(frame)

            cv2.imshow("YOLOv8 + Telemetry (MAVSDK)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        stop_flag = True
        telemetry_task.cancel()
        time.sleep(1)  # give time for async task to end

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        loop.call_soon_threadsafe(loop.stop)
        t.join()
        print("[INFO] Clean exit. Video saved to:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
