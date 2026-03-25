#2 = nose

import detection_code as LM
import cv2 as cv

cap = cv.VideoCapture(0)  # 0 as a default webcam
face_detector = LM.Face()  # detect face with variable

picture_path = '/Users/koviressler/Desktop/DailyTefillin/eye_detection/tefillinEx.jpeg'

def draw_nose_points(img, detected_nose):
    if detected_nose != None:
        cv.circle(img, (detected_nose[1], detected_nose[2]), 3, (0, 255, 0), -1)

def detect_nose(image):
    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    face_detector.Find_Points(image, imgRGB)

    nose_points = face_detector.Nose_Bottom()

    return nose_points


# img = cv.imread(picture_path)

# # Detect eyes
# detected_eyes = detect_nose(img)

# # Draw eye points on the image
# draw_nose_points(img, detected_eyes)

# # Display the image
# cv.imshow("img", img)
# cv.waitKey(0)  # Wait for a key press
# cv.destroyAllWindows()


while True:
    ret, img = cap.read()
    if not ret:
        break
    
    draw_nose_points(img, detect_nose(img)) #front camera

    cv.imshow("img", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()