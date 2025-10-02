from ultralytics import YOLO # type: ignore
import cv2

INDEX_CAM = 1
# 1: Sử dụng camera từ điện thoại thông qua ứng dụng ivCam
# 0: Sử dụng camera từ máy tính

cap = cv2.VideoCapture(1)

model = YOLO("Model/best_v4.pt")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, show=False, conf=0.6, verbose = False)
    annotated_frame = results[0].plot()
    cv2.imshow("Fire Detection", annotated_frame)
    for r in results:
        for c in r.boxes.cls:
            if int(c) == 0:
                print("CẢNH BÁO: Phát hiện đám cháy!")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
