import os

print("Chọn chế độ:")
print("1. Chụp ảnh khuôn mặt")
print("2. Huấn luyện mô hình")
print("3. Nhận diện khuôn mặt")


while True:
    choice = input("Nhập số: ")
    if choice == "1":
        os.system("python src/capture_faces.py")
    elif choice == "2":
        os.system("python src/train_face_recognition.py")
    elif choice == "3":
        os.system("python src/recognition.py")
    else:
        print("Lựa chọn không hợp lệ!")
