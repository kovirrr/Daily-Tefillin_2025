#========== IMPORTS =============
import cv2
import numpy as np
from ultralytics import YOLO
import detection_code as LM #local file
import time


#========== VARIABLES =============
face_detector = LM.Face()#detects face with variables 
#========== FUNCTIONS =============
def read(image_path):
    return cv2.imread(image_path)

def detect_eyes(image):
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_detector.Find_Points(image, imgRGB)

    left_eye_points = face_detector.Left_Eye()
    right_eye_points = face_detector.Right_Eye()

    if not left_eye_points or not right_eye_points:
        return None
    
    return [left_eye_points[1], right_eye_points[0]] # Return only the inner corners of each eye
    #format of each is [id, x, y]

def detect_face(image): #helper func for face_features
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pts = face_detector.Find_Points(image, imgRGB)
    if pts == []:
        return None 
    return pts

def detect_hairline(image):
    model = YOLO(
        '/Users/koviressler/Desktop/DailyTefillin/'
        'hairline_detection/hairlineAI.pt'
    )
    results = model(image)  # ultralytics Results object
    r = results[0]
    if r.masks is None or len(r.masks.xy) == 0:
        return None

    confs = r.boxes.conf.cpu().numpy()
    # pick the index of the highest-confidence detection
    best_i = int(confs.argmax())

    # r.masks.xy is a list of Nx2 point-arrays, one per detection
    best_mask = r.masks.xy[best_i]
    # convert to a flat list of [x,y]
    return best_mask.tolist()

def detect_tefillin(image):
    model = YOLO(
        '/Users/koviressler/Desktop/DailyTefillin/'
        'tefillin_detection/runs/detect/train6/weights/best.pt'
    )
    results = model(image)  # ultralytics Results object
    r = results[0]          # single-image batch
    confs = r.boxes.conf.cpu().numpy()
    if len(confs) == 0:
        return []

    #highest confidence
    best_i = int(confs.argmax())
    x1, y1, x2, y2 = r.boxes.xyxy[best_i].cpu().numpy()

    # return its four corners
    return [
        [x1, y1],
        [x1, y2],
        [x2, y1],
        [x2, y2],
    ]

def lowest_hairline_point(eye_cords, forehead_cords):
    print("Eye Coordinates:", eye_cords)
    if not eye_cords[0] or not eye_cords[1] or not forehead_cords:
        return None
    left_x = eye_cords[0][1]  # right-most x value of left eye
    right_x = eye_cords[1][0]  # left-most x value of right eye

    imp_points = []
    for i in range(len(forehead_cords)): #only take points between eyes. we don't care about side hair, and this accounts for widows peak
        if left_x < forehead_cords[i][0] < right_x:
            imp_points.append(forehead_cords[i])

    #artifficially add in ideal point (if they don't have widows peak)
    forehead_cords.sort(key=lambda x: x[1])  # Sort by y value
    left_most_point = None
    right_most_point = None
    for point in forehead_cords: #from highest y-value to lowest
        if point[0] < left_x and not left_most_point: #highest point on left side of face (doesn't include points between eyes)
            left_most_point = point
        elif point[0] > right_x and not right_most_point:
            right_most_point = point
        if left_most_point and right_most_point:
            break
        
    #find midpoint y val
    if left_most_point and right_most_point:
        artif_point = [((left_x + right_x) / 2), ((left_most_point[1] + right_most_point[1]) / 2)]
        imp_points.append(artif_point) #average of both

    if not imp_points:
        return None
    lowest_point = max(imp_points, key=lambda p: p[1])
    return lowest_point  # Return the lowest point on the hairline between the eyes + the artifical point if needed

def get_lowest_hairline_point(image):
    
    eye_coordinates = detect_eyes(image)
    hairline_points = detect_hairline(image)

    # Find the lowest hairline point
    lowest_point = lowest_hairline_point(eye_coordinates, hairline_points)
    return lowest_point  # Return the lowest hairline point found in the image

def face_features(image): #put in a cv2.imread(frame)
    pnts = detect_face(image)
    if pnts:
        return [pnts[152], pnts[2]]#chin, nose tip
    return [None, None]  # Return None if no points are found

def detect_head_pose(image):
    face_landmarks = detect_face(image)
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        (face_landmarks[30][1:3]),     # Nose tip
        (face_landmarks[152][1:3]),    # Chin
        (face_landmarks[36][1:3]),     # Left eye left corner
        (face_landmarks[45][1:3]),     # Right eye right corner
        (face_landmarks[48][1:3]),     # Left Mouth corner
        (face_landmarks[54][1:3])      # Right mouth corner
    ], dtype="double")

    # Camera internals
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    return euler_angles #returns a list of 3 lists

def compare_poses(current_pose, reference_pose, threshold=4.0): #returns the biggest difference in head position (how to fix it)
    pitch_diff = current_pose[0][0] - reference_pose[0][0] #left + right
    yaw_diff = current_pose[1][0] - reference_pose[1][0]
    roll_diff = current_pose[2][0] - reference_pose[2][0]
    
    guidance = []
    if abs(pitch_diff) > threshold:
        guidance.append(("Turn " + ("right" if pitch_diff > 0 else "left"), abs(pitch_diff)))
    if abs(roll_diff) > threshold:
        guidance.append(("Tilt head " + ("left" if roll_diff > 0 else "right"), abs(roll_diff)))
    
    # Sort guidance by magnitude of difference
    guidance.sort(key=lambda x: x[1], reverse=True)
    
    if abs(yaw_diff) > threshold:
        guidance.insert(0, ("Turn " + ("down" if yaw_diff > 0 else "up"), abs(yaw_diff)))

    return guidance

def distance(point1, point2): #gets dis between 2 points - list of [x,y,...]
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def initialize(ref_image): #ref_pos
    global ref_pos, ref_bottom_dis, ref_top_dis, initialized
    initialized = True
    ref_hairline = detect_hairline(ref_image)
    ref_eyes = detect_eyes(ref_image)
    ref_hairpoint = lowest_hairline_point(ref_eyes, ref_hairline)
    ref_features = face_features(ref_image) #0 = chin, 1 = nose

    #global #1
    ref_pos = detect_head_pose(ref_image)

    if ref_pos is None or ref_features is None:
        return None
    
    ref_bottom_dis = distance(ref_features[0][1:3], ref_features[1][1:3]) #distance from chin to nose
    ref_top_dis = distance(ref_features[1][1:3], ref_hairpoint) #distance from nose to forehead point


def calc_hairline(img):
    pose_diff = compare_poses(detect_head_pose(img), ref_pos)
    if pose_diff == []: #they are in range
        feat = face_features(img) #[[chin id,x,y],[nose id,x,y]]
        if feat != [None, None]:
            bottom_dis = distance(feat[0][1:3], feat[1][1:3])
            return feat[1][2] - (ref_top_dis*bottom_dis/ref_bottom_dis)
        return "Can't detect chin + nose"
    print(pose_diff)
    return "Head not in position"


def draw_point_on_image(image, point, color=(0, 0, 255), radius=5, window_name="Point"): #testing purposes only
    if point is None or len(point) < 2:
        print("Invalid point provided.")
        return

    x = int(point[0])
    y = int(point[1])

    img_copy = image.copy()
    cv2.circle(img_copy, (x, y), radius, color, thickness=-1)
    cv2.imshow(window_name, img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tef_good(image, debug=False): #input cv2.imread(frames) of the pic
    if not initialized:
        return [False, "Need reference image"]
    eye_cords = detect_eyes(image) #the two inner corners of the eyes
    hairline_point = calc_hairline(image)
    tef_cords = detect_tefillin(image) #4 points of tefillin box
    if not eye_cords or type(eye_cords) is not list:
        return [False, "No eyes detected"]
    if not hairline_point or type(hairline_point) is str:
        return [False, hairline_point] #error message
    if not tef_cords or type(tef_cords) is not list:
        return [False,"No tefillin detected"]
    teffilin_point = [((tef_cords[0][0] + tef_cords[2][0]) / 2), tef_cords[1][1]] #lowest point of box (y), in the middle (x)

    if debug:
        img_copy = image.copy()
        # Draw tefillin box corners (blue)
        for pt in tef_cords:
            cv2.circle(img_copy, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)
        # Draw teffilin_point (center bottom of box, cyan)
        cv2.circle(img_copy, (int(teffilin_point[0]), int(teffilin_point[1])), 7, (255, 255, 0), -1)
        # Draw eye points (green)
        for pt in eye_cords:
            cv2.circle(img_copy, (int(pt[1]), int(pt[2])), 7, (0, 255, 0), -1)
        # Draw hairline as a horizontal line (red)
        cv2.line(img_copy, (0, int(hairline_point)), (img_copy.shape[1], int(hairline_point)), (0, 0, 255), 2)
        chin_nose = face_features(image)
        if chin_nose and chin_nose[0] and chin_nose[1]:
            cv2.circle(img_copy, (int(chin_nose[0][1]), int(chin_nose[0][2])), 7, (255, 0, 255), -1) # Chin (magenta)
            cv2.circle(img_copy, (int(chin_nose[1][1]), int(chin_nose[1][2])), 7, (0, 255, 255), -1) # Nose (yellow)
        cv2.imshow("Debug Tefillin Placement", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if teffilin_point[0] < eye_cords[0][1]:
        return [False, "Too far left"]
    if teffilin_point[0] > eye_cords[1][1]:
        return [False, "Too far right"]
    if teffilin_point[1] > hairline_point:
        return [False, "Too low"]
    return [True, ""]


    

#========= USAGE ===========

p = "/Users/koviressler/Daily-Tefillin_2025/tefillin_detection/finallyTef/train/images/17dae0c8-139a-4c0d-aa5e-c65f815e374c 2.JPG"
z = "/Users/koviressler/Desktop/DailyTefillin/people/zacky.JPG"

#kovi testing
blank = read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_blank.JPG")
blank2 = read("/Users/koviressler/Daily-Tefillin_2025/people/Screenshot 2025-10-01 at 3.50.12 PM.png")


left = read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_left.JPG")
right = read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_right.JPG")
low = read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_low.JPG")
good = read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_good.JPG")

initialize(blank)


# print(tef_good(left, ref=blank, debug=True))
# print(tef_good(right, ref=blank, debug=True))
#print(tef_good(low, debug=True))
initialize(blank2)

print(tef_good(good, debug=True))