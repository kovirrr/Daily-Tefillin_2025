import cv2
import numpy as np
from ultralytics import YOLO
import time

def detect_hairline_realtime(model_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Initialize the front camera
    cap = cv2.VideoCapture(0)  # 0 is usually the front camera. If it doesn't work, try 1.

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Perform inference
        results = model(frame, conf=.75)

        if len(results) > 0 and results[0].masks is not None:
            # Get the mask with the highest confidence
            # best_mask = results[0].masks.xy[0]
            
            # # Convert to integer coordinates
            # best_mask = best_mask.astype(np.int32)

            # Draw the polygon
            for mask in results:
                best_mask = results[0].masks.xy[0]
                best_mask = best_mask.astype(np.int32)
                cv2.polylines(frame, [best_mask], isClosed=True, color=(0, 255, 0), thickness=2)
            
            #print(f"Detected points: {best_mask.tolist()}")

        # Display the resulting frame
        cv2.imshow('Hairline Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
def detect_hairline_from_image(model_path, image_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read image.")
        return

    # Perform inference
    results = model(image, conf=0.75)

    if len(results) > 0 and results[0].masks is not None:
        for mask in results:
            best_mask = results[0].masks.xy[0]
            best_mask = best_mask.astype(np.int32)
            cv2.polylines(image, [best_mask], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the image with detections
    cv2.imshow('Hairline Detection (Image)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
model_path = '/Users/koviressler/Desktop/DailyTefillin/hairline_detection/hairlineAI.pt'

#detect_hairline_realtime(model_path)
detect_hairline_from_image(model_path, "/Users/koviressler/Desktop/DailyTefillin/people/zacky.JPG")