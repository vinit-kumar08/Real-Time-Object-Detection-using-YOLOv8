from ultralytics import YOLO

# Load the YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Path of the image
image_path = "test.jpg"

# Perform object detection
results = model.predict(
    source=image_path,
    conf=0.5,
    save=True,
    show=True
)

print("Object detection completed successfully!")