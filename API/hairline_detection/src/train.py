from ultralytics import YOLO

def train_yolo_model():
    # Load a model
    model = YOLO('yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model

    # Train the model
    results = model.train(
        data='/Users/koviressler/Desktop/DailyTefillin/hairline_detection/data.yaml',
        epochs=100,
        imgsz=640,
        patience=50,
        batch=16,
        save=True
    )

    # Export the model to ONNX format
    model.export(format='onnx')

if __name__ == "__main__":
    train_yolo_model()