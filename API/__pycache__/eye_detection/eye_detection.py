#left eye edge = 243
#right eye edge = 463

import detection_code as LM
import cv2 as cv
import sys

cap = cv.VideoCapture(0)  # 0 as a default webcam
face_detector = LM.Face()  # detect face with variable

picture_path = '/Users/koviressler/Desktop/DailyTefillin/people/zacky.JPG'

def draw_eye_points(img, detected_eyes):
    if detected_eyes != [None, None]:
        for i in range(2):
            for point in detected_eyes[i]:
                # Draw a circle at each eye point
                cv.circle(img, (point[1], point[2]), 3, (0, 255, 0), -1)
                print(detected_eyes[i])  # Print the list of points

def detect_eyes(image):
    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    face_detector.Find_Points(image, imgRGB)

    left_eye_points = face_detector.Left_Eye()
    right_eye_points = face_detector.Right_Eye()

    return [left_eye_points, right_eye_points]


img = cv.imread(picture_path)

# Detect eyes
detected_eyes = detect_eyes(img)

# Draw eye points on the image
draw_eye_points(img, detected_eyes)

# Display the image
cv.imshow("img", img)
cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()


while True:
    ret, img = cap.read()
    if not ret:
        break
    
    draw_eye_points(img, detect_eyes(img)) #front camera

    cv.imshow("img", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()