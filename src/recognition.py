import cv2
import os
import dlib
import numpy as np
import face_recognition
import pickle
from liveness_detection import detect_liveness

# Khởi tạo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encodings_path = os.path.join(BASE_DIR, "..", "models", "encodings.pickle")
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

prev_gray, prev_name = None, None
blink_detected, movement_detected = False, False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Kiểm tra giả mạo
    liveness_status, liveness_color, blink_detected, movement_detected, prev_gray = detect_liveness(
        frame, prev_gray, prev_name, blink_detected, movement_detected
    )
    
    faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    name = "Unknown"
    
    for face in faces:
        try:
            face_encodings = face_recognition.face_encodings(
                rgb_frame, [(face.top(), face.right(), face.bottom(), face.left())]
            )
            if face_encodings:
                face_encoding = face_encodings[0]
                face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:
                    name = data["names"][best_match_index]
        except Exception as e:
            print(f"WARNING: Face encoding error: {e}")
        
        if name != prev_name:
            blink_detected, movement_detected = False, False
            prev_name = name
            liveness_status = "Analyzing..."
            liveness_color = (255, 255, 0)
        
        cv2.putText(frame, f"Person: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, liveness_status, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, liveness_color, 2)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition + Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()