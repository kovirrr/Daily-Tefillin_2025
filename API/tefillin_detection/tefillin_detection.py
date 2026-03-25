import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/runs/detect/train6/weights/best.pt') #tefillin detection model

def detect_tefillin(image):
    # Run inference on the image
    results = model(image)
    
    # Process and draw results
    for result in results: #for every tefillin detected
        boxes = result.boxes.xyxy.cpu().numpy() #set boxes = a list of the coordinates
        confidences = result.boxes.conf.cpu().numpy() #set this = the confidence
        
        for box, confidence in zip(boxes, confidences): #for each coordinate in boxes
            if confidence > 0.7:  #(can change confidence)
                x1, y1, x2, y2 = map(int, box[:4]) #set variables equal to the 4 values
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                cv2.putText(image, f'Tefillin: {confidence:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

def test_on_image(image_path): #tries to run the code on an miage
    image = cv2.imread(image_path) #try accessing image
    
    if image is None: #if image isn't there
        raise ValueError(f"Unable to read image at {image_path}")

    results= detect_tefillin(image)
    cv2.imshow('Tefillin Detection', results) #display results continously
    cv2.waitKey(0) #if they press a letter it exits
    cv2.destroyAllWindows()
    return results

# Test on webcam feed
def test_on_webcam():
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    while True:
        ret, frame = cap.read() #keep reading the frame
        if not ret:
            break
        
        result_frame = detect_tefillin(frame)
        cv2.imshow('Tefillin Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): #if they press q, exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Choose which test to run
test_on_webcam()  # Comment this out if you want to test on a single image instead
#test_on_image('/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/all_tefillin/ALLphotos/PHOTO-2024-08-27-09-52-02.jpg')