import cv2
import numpy as np
import dlib
from math import hypot
from pathlib import Path

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = Path(__file__).resolve().parent / "shape_predictor_68_face_landmarks.dat"
predictor = None


def load_predictor():
    """Load the external landmark model on first use."""
    global predictor
    if predictor is None:
        if not PREDICTOR_PATH.is_file():
            raise FileNotFoundError(
                f"Missing landmark model: {PREDICTOR_PATH}. See README.md for setup."
            )
        predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    return predictor

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length if ver_line_length else float("inf")
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks, frame, gray):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)


    cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
    mask = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, (0,0,255), 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = max(0, int(np.min(left_eye_region[:, 0])))
    max_x = min(gray.shape[1], int(np.max(left_eye_region[:, 0])))
    min_y = max(0, int(np.min(left_eye_region[:, 1])))
    max_y = min(gray.shape[0], int(np.max(left_eye_region[:, 1])))

    if max_x <= min_x or max_y <= min_y:
        return None, (0, 0)
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(round((width / 2)))]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(round(width / 2)): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 0.4
    elif right_side_white == 0:
        gaze_ratio = 4
    else:
        gaze_ratio = left_side_white / right_side_white


    eye_center = (min_x + max_x)//2,(min_y + max_y) // 2

    return gaze_ratio, eye_center

def process_gaze_frame(frame):
    landmark_predictor = load_predictor()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_info = {"direction": "None","left_eye_center":(0,0),"right_eye_center":(0,0)}

    for face in faces:
        landmarks = landmark_predictor(gray, face)

        # Blinking detection
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 0))

        # Gaze detection
        gaze_ratio_left_eye,left_eye_center = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, frame, gray)
        gaze_ratio_right_eye, right_eye_center = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, frame, gray)
        if gaze_ratio_left_eye is None or gaze_ratio_right_eye is None:
            continue
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        gaze_info["left_eye_center"] = left_eye_center
        gaze_info["right_eye_center"] = right_eye_center

        if gaze_ratio <= 1:
            gaze_info["direction"] = "Right"
            cv2.putText(frame, "RIGHT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2.6, (0, 0, 255), 3)
        elif 1 < gaze_ratio < 3:
            gaze_info["direction"] = "Center"
            cv2.putText(frame, "CENTER", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2.6, (0, 0, 255), 3)
        else:
            gaze_info["direction"] = "Left"
            cv2.putText(frame, "LEFT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2.6, (0, 0, 255), 3)

    return frame, gaze_info

def release_resources():
    cv2.destroyAllWindows()
