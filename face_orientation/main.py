import cv2
import numpy as np
from ultralytics import YOLO
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import detection_code as LM

face_detector = LM.Face()
def detect_face(image):
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pts = face_detector.Find_Points(image, imgRGB)
    if pts == []:
        return None
    return pts
def detect_head_pose(image):
    face_landmarks = detect_face(image)
    if face_landmarks is None:
        raise Exception
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    image_points = np.array([
        (face_landmarks[30][1:3]),
        (face_landmarks[152][1:3]),
        (face_landmarks[36][1:3]),
        (face_landmarks[45][1:3]),
        (face_landmarks[48][1:3]),
        (face_landmarks[54][1:3])
    ], dtype="double")
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return euler_angles
def compare_poses(current_pose, reference_pose, threshold=4.0):
    pitch_diff = current_pose[0][0] - reference_pose[0][0]
    yaw_diff = current_pose[1][0] - reference_pose[1][0]
    roll_diff = current_pose[2][0] - reference_pose[2][0]
    guidance = []
    if abs(pitch_diff) > threshold:
        guidance.append(("Turn " + ("right" if pitch_diff > 0 else "left"), abs(pitch_diff)))
    if abs(roll_diff) > threshold:
        guidance.append(("Tilt head " + ("left" if roll_diff > 0 else "right"), abs(roll_diff)))
    guidance.sort(key=lambda x: x[1], reverse=True)
    if abs(yaw_diff) > threshold:
        guidance.insert(0, ("Turn " + ("down" if yaw_diff > 0 else "up"), abs(yaw_diff)))
    return guidance
ref_img = cv2.imread('/Users/koviressler/Daily-Tefillin_2025/people/kovi_good.JPG')
ref_pose = detect_head_pose(ref_img)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    try:
        guidance = compare_poses(detect_head_pose(frame), ref_pose)
    except: guidance = [("Face not detected", 0)]
    cv2.putText(frame, 'Guidance:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    for i, (text, val) in enumerate(guidance):
        cv2.putText(frame, f'{text}: {val:.2f}', (10, 60+30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imshow('Face Orientation Guidance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
