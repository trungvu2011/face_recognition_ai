import cv2
import os
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
detector = dlib.get_frontal_face_detector()
predictor = os.path.join(BASE_DIR, "..", "models", "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor)

# Ngưỡng phát hiện
BLINK_THRESHOLD = 0.22
MOVEMENT_THRESHOLD = 0.2

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_liveness(frame, prev_gray, prev_name, blink_detected, movement_detected):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return "No Face Detected", (0, 255, 255), False, False, prev_gray
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        leftEye, rightEye = shape[36:42], shape[42:48]
        leftEAR, rightEAR = eye_aspect_ratio(leftEye), eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        if prev_name is None:
            blink_detected, movement_detected = False, False
        
        if ear < BLINK_THRESHOLD:
            blink_detected = True
        
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            avg_movement = np.mean(cv2.cartToPolar(flow[..., 0], flow[..., 1])[0])
            
            if avg_movement > MOVEMENT_THRESHOLD:
                movement_detected = True
        
        prev_gray = gray.copy()
        
        if blink_detected and movement_detected:
            return "Real Person", (0, 255, 0), True, True, prev_gray
        else:
            return "Fake", (0, 0, 255), blink_detected, movement_detected, prev_gray
    
    return "Analyzing...", (255, 255, 0), blink_detected, movement_detected, prev_gray