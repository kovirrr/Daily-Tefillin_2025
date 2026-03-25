from ultralytics import YOLO

# Define the path to your dataset YAML file
DATASET_PATH = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/finallyTef/data.yaml"

# Load the YOLO model (using a pretrained model for transfer learning)
model = YOLO("yolov8n.pt")  # You can also try 'yolov8s.pt' for better accuracy

# Train the model
model.train(
    data=DATASET_PATH,  # Path to the YAML file
    epochs=400,          # Adjust the number of epochs as needed
    imgsz=640,          # Image size for training
    batch=16,            # Batch size (adjust based on your GPU memory)
    patience= 100,
    device="cpu"       # Use "cpu" if you don't have a GPU
)

# Save the trained model
model_path = "runs/detect/train/weights/best.pt"
print(f"Model trained successfully! Saved at {model_path}")
