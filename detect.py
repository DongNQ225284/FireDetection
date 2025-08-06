from ultralytics import YOLO # type: ignore
import cv2

# Load mô hình
model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=False, conf=0.5)

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
