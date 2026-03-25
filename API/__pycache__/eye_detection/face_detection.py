#bottom of nose = 2
import detection_code as LM
import cv2 as cv

cap = cv.VideoCapture(0)  # 0 as a default webcam
face_detector = LM.Face()  # detect face with variable

pic = '/Users/koviressler/Desktop/DailyTefillin/eye_detection/tefillinEx.jpeg'

img = cv.imread(pic)

if img is None:
    print("Error: Could not read the image.")
else:
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    face_detector.Drawing(img, imgRGB, color=(0, 110, 100), thickness=3, circle_radius=3)
    face_detector.Find_Points(img, imgRGB)


    cv.imshow("img", img)
    cv.waitKey(0)  # Wait until a key is pressed


while True:
    ret, img = cap.read()
    if not ret:
        break
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    face_detector.Drawing(img, imgRGB, color=(0, 110, 100), thickness=3, circle_radius=3)
    face_detector.Find_Points(img, imgRGB)

    #print(face_detector.Left_Eye()) #returns a list of points

    cv.imshow("img", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()