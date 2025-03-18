import os
import pickle
import cv2
import face_recognition
import dlib

# Load mô hình CNN của dlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cnn_model_path = os.path.join(BASE_DIR, "..", "models", "mmod_human_face_detector.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)

# Thư mục chứa dataset hình ảnh
dataset_path = os.path.join(BASE_DIR, "..", "dataset")
encodings_path = os.path.join(BASE_DIR, "..", "models", "encodings.pickle")

# Danh sách để lưu tên và mã hóa khuôn mặt
known_encodings = []
known_names = []

# Duyệt qua từng thư mục con (mỗi thư mục là một người)
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    # Kiểm tra nếu không phải thư mục thì bỏ qua
    if not os.path.isdir(person_folder):
        continue

    print(f"🔄 Đang xử lý: {person_name}")

    # Duyệt qua từng file ảnh trong thư mục
    for filename in os.listdir(person_folder):
        image_path = os.path.join(person_folder, filename)

        # Đọc ảnh và chuyển đổi sang RGB
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300, 300))

        if image is None:
            print(f"⚠️ Không thể đọc ảnh: {image_path}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

       # Phát hiện khuôn mặt bằng CNN
        face_detections = cnn_face_detector(rgb_image, 1)
        if len(face_detections) == 0:
            print(f"🚫 Không tìm thấy khuôn mặt trong ảnh: {image_path}")
            continue

        # Lấy vị trí khuôn mặt từ kết quả CNN
        face_locations = [(d.rect.top(), d.rect.right(), d.rect.bottom(), d.rect.left()) for d in face_detections]

        # Mã hóa khuôn mặt (lấy khuôn mặt đầu tiên)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Lưu dữ liệu nếu có khuôn mặt hợp lệ
        if len(face_encodings) > 0:
            known_encodings.append(face_encodings[0])
            known_names.append(person_name)

# Kiểm tra nếu không có dữ liệu nào được mã hóa
if len(known_encodings) == 0:
    print("❌ Không có khuôn mặt nào được mã hóa. Hãy kiểm tra lại dataset!")
    exit()

# Lưu dữ liệu mã hóa vào file encodings.pickle
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_path, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Đã lưu mã hóa khuôn mặt vào: {encodings_path}")
