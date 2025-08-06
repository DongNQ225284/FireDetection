from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="data/data.yaml", epochs=100, imgsz=416)
