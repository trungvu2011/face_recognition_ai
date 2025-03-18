import cv2
import os
import pickle
import face_recognition
import numpy as np

# Đường dẫn lưu dataset và model
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../dataset")
ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), "../models/encodings.pickle")

# Tạo thư mục dataset nếu chưa có
os.makedirs(DATASET_PATH, exist_ok=True)

# Nhập tên người dùng
name = input("Nhập tên của bạn: ").strip()
user_path = os.path.join(DATASET_PATH, name)
os.makedirs(user_path, exist_ok=True)

# Load dữ liệu nhận diện khuôn mặt hiện có
if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}

# Khởi động camera
cap = cv2.VideoCapture(0)
count = len(os.listdir(user_path))  # Đếm số ảnh đã có trong thư mục

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    # Chụp ảnh khi nhấn phím "s"
    if key == ord("s"):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            count += 1
            top, right, bottom, left = face_locations[0]
            face_image = frame[top:bottom, left:right]

            img_path = os.path.join(user_path, f"{count}.jpg")
            cv2.imwrite(img_path, face_image)
            print(f"📸 Ảnh {count} đã lưu!")

            # Mã hóa khuôn mặt
            face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])
            if face_encodings:
                data["encodings"].append(face_encodings[0])
                data["names"].append(name)
                print("✅ Khuôn mặt đã được mã hóa và thêm vào dữ liệu!")

        else:
            print("❌ Không tìm thấy khuôn mặt, vui lòng thử lại!")

    # Thoát khi nhấn phím "q"
    elif key == ord("q"):
        break

# Lưu dữ liệu mới vào file encodings.pickle
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)
    print("💾 Dữ liệu nhận diện đã được cập nhật!")

cap.release()
cv2.destroyAllWindows()
