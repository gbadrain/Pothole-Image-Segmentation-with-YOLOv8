from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model

# Train the model
results = model.train(data='/Users/GURU/pothole-segmentation/data/data.yaml', epochs=100, imgsz=640)
