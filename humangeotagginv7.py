import cv2
import os
from datetime import datetime

# Create a directory for detections if it doesn’t exist
output_dir = "detections"
os.makedirs(output_dir, exist_ok=True)

# Confidence threshold for saving
SAVE_CONF_THRESHOLD = 0.6   # change as needed

# ---------------------------
# Inside your YOLO detection loop
# ---------------------------

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()

    # Flag to decide if we save this frame
    save_frame = False

    for box, conf, cls in zip(boxes, confs, clss):
        if int(cls) == 0 and conf >= SAVE_CONF_THRESHOLD:  
            # Person detected with enough confidence
            save_frame = True  

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # If we should save → overlay GPS and write image
    if save_frame:
        telemetry_text = f"Lat: {latitude:.6f}, Lon: {longitude:.6f}, Alt: {altitude:.2f}"
        cv2.putText(frame, telemetry_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"person_{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved frame -> {filename}")
