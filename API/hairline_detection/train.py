from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    patience=20,
    batch=16,
    save=True
)

# Export the model to ONNX
model.export(format='onnx')