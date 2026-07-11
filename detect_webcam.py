from ultralytics import YOLO
import cv2
import time
from collections import Counter
import csv
import os
from datetime import datetime

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Get webcam properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = 20

# Video output path
video_path = "outputs/videos/detection_video.mp4"

# Codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Video Writer
out = cv2.VideoWriter(
    video_path,
    fourcc,
    fps_video,
    (frame_width, frame_height)
)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

prev_time = 0

os.makedirs("outputs/videos", exist_ok=True)

os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/screenshots", exist_ok=True)

log_file = "outputs/logs/detection_log.csv"

# Create CSV file if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Object", "Confidence"])

while True:

    success, frame = cap.read()

    if not success:
        break

    # Object Detection
    results = model(
        frame,
        imgsz=640,
        conf=0.5,
        verbose=False
    )

    # Save detections to CSV
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[cls]
            current_time = datetime.now().strftime("%H:%M:%S")

            writer.writerow([
                current_time,
                object_name,
                round(confidence, 2)
            ])

    # Count detected objects
    names = model.names
    detected_objects = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        detected_objects.append(names[cls])

    counts = Counter(detected_objects)

    # Draw detections
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    # Display object counts
    y = 80

    for obj, count in counts.items():
        cv2.putText(
            annotated_frame,
            f"{obj}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )
        y += 30

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'S' to save screenshot
    if key == ord("s"):
        filename = f"outputs/screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"Screenshot saved: {filename}")

    # Press 'Q' to quit
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
out.release()