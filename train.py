from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO("runs/detect/train5/weights/last.pt")
    model.train(data="Fire_indoor_data/data.yaml", batch = 25, epochs=100, imgsz=416)
    