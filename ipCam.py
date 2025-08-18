import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} hoạt động")
        cap.release()

cap = cv2.VideoCapture(1)  # Thay 0 bằng index tìm được ở bước 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được hình ảnh từ iVCam!")
        break

    cv2.imshow("iPhone Camera via iVCam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Bấm 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
