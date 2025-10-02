from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO("model/yolo11n.pt")
    model.train(
        data="data/Fire_indoor_data_v3/data.yaml",  # file dataset
        batch=16,                                   # giảm batch khi tăng imgsz
        epochs=300,                                 # fine-tune thêm
        imgsz=640,                                  # tăng độ phân giải ảnh
        lr0=0.001,                                  # learning rate nhỏ hơn
        augment=True,                               # bật augmentation
        patience=0,                                 # early stopping nếu không cải thiện
        degrees=0.0,                                # augmentation
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1
    )
    