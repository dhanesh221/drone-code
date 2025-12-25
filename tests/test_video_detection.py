import cv2
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("best.pt")  # Change the model path if necessary

# Use the RTSP URL you found
video_stream_url = "rtsp://192.168.144.25:8554/main.264"  # Replace with your camera's RTSP URL

# Open the video stream
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("Error: Unable to access the video stream.")
    exit()

# Get the class labels from YOLO model
class_names = model.names  # YOLOv8 provides a 'names' attribute that maps class indices to labels

# Define the codec and create a VideoWriter object to save the video
output_file = 'output_detected_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec (mp4)
frame_rate = 30  # FPS, you can adjust it depending on your stream's FPS
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

while True:
    # Capture each frame from the video feed
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read the frame.")
        break

    # Run object detection on the current frame
    results = model(frame)  # Run detection

    # Extract the detected boxes and labels
    boxes = results[0].boxes  # Get the bounding boxes (as a tensor)

    detected = False  # Flag to check if any objects are detected

    # Loop through the detected boxes and draw them on the frame
    for box in boxes:
        # Unpack box coordinates (x1, y1, x2, y2) and confidence
        x1, y1, x2, y2 = box.xyxy[0]  # Top-left and bottom-right points
        confidence = box.conf[0]  # Confidence score
        label = int(box.cls[0])  # Class index
        class_name = class_names[label]  # Get class label

        # Only proceed if the confidence is above a threshold (e.g., 0.5)
        if confidence > 0.5:
            detected = True  # Mark that something has been detected

            # Convert to int for drawing the rectangle
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
            # Add the label (class name) and confidence score
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed frame to the video file if any objects were detected
    if detected:
        out.write(frame)

    # Show the frame with detection boxes
    cv2.imshow("Object Detection", frame)

    # Break on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
out.release()  # Release the VideoWriter
cv2.destroyAllWindows()

print("Video saved successfully.")
