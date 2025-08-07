from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO("runs/detect/train4/weights/last.pt")
    model.train(data="Fire_indoor_data/data.yaml", batch = 20, epochs=100, imgsz=416)
    