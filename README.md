# Mô tả project

Project này trình bày hướng dẫn sử dụng YOLO để phát hiện đám cháy trong nhà từ **camera** trong thời gian thực.

# Quy trình thực hiện

## Bước 1: Chuẩn bị dữ liệu

- **Những nhãn xuất hiện là gì?**
- **Nguồn ảnh cần như thế nào?**

### Xây dựng nhãn

**Nhãn được sử dụng trong Project:**

**Version 1**

1. `fire`: cháy thật.

**Version 2**

1. `fire`: cháy thật.
2. `non-fire`: các đối tượng dễ bị nhầm thành lửa.

### Thu thập dữ liệu

#### Chất lượng dữ liệu

- Độ phân dải cao.
- Hình ảnh rõ ràng, ít nhiễu hoặc mờ quá mức.

#### Sự đa dạng về dữ liệu

- Đa dạng về môi trường: ngoài trời, trong nhà, nhiều loại cảnh.
- Đa dạng về điều kiện ánh sáng: ban ngày, ban đêm, ánh sáng yếu, ánh sáng mạnh.
- Đa dạng về hình dạng và kích thước đối tượng: cháy nhỏ, cháy lớn, cháy bị che khuất, khói, đám lửa lan rộng.
- Đa dạng về background: tránh model học nhầm pattern của nền thay vì lửa.

#### Cân bằng dữ liệu

- Số lượng ảnh có cháy và không cháy tương đối cân bằng.

- Nếu **imbalance**, cần xử lý bằng **augmentation** hoặc **weighting loss**.

### Nguồn dữ liệu được sử dụng

Nguồn ảnh được lấy trên [Roboflow Universe](https://universe.roboflow.com) và được đánh nhãn lại.

### Đánh nhãn

- Sử dụng [Roboflow](https://universe.roboflow.com) để đánh nhãn, xử lý dữ liệu.

- Đánh nhãn chính xác: **bounding box** cần bao trọn đối tượng

#### **Kết quả thực hiện**

**Version 1: [Fire Indoor v1](https://universe.roboflow.com/nguyen-dong-ys7mf/fire-indoor-3rnk5/dataset/3)**
| Nhãn | Số lượng nhãn |
|------|---------------|
|`fire`| 1499 |

**Version 2: [Fire Indoor v2](https://universe.roboflow.com/nguyen-dong-ys7mf/fire-indoor-3rnk5/dataset/2)**
| Nhãn | Số lượng nhãn |
|------|---------------|
|`fire`| 1499 |
|`non-fire`| 702 |

## Bước 2: Kiến trúc model

### YOLOv5

### YOLOv8

### YOLOv11

Project sử dụng YOLOv5, YOLOv8, YOLOv11

## Bước 3: Huấn luyện

Đầu vào là tập dữ liệu được đánh nhãn, cấu trúc dữ liệu chuẩn YOLO.

```yaml
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
```

Các File trong folder labels có định dạng `.txt`:
Mỗi dòng trong file `.txt` tương ứng với 1 object:

```yaml
<class_id> <x_center> <y_center> <width> <height>
```

Trong folder của dự án file `train.py` chứa code để train mô hình. Cần config tham số truyền vào (hyperparameter): learning rate, batch size, epoch, optimizer, ....

| Nhóm                  | Hyperparameter    | Ý nghĩa                    |
| --------------------- | ----------------- | -------------------------- |
| **Huấn luyện**        | `epochs`          | Số vòng lặp huấn luyện     |
|                       | `batch_size`      | Số ảnh trong 1 batch       |
|                       | `imgsz`           | Kích thước ảnh đầu vào     |
|                       | `workers`         | CPU workers cho dataloader |
| **Tối ưu hóa**        | `optimizer`       | Thuật toán tối ưu          |
|                       | `lr0`             | Learning rate ban đầu      |
|                       | `lrf`             | Learning rate cuối (decay) |
|                       | `momentum`        | Momentum cho SGD           |
|                       | `weight_decay`    | Regularization             |
|                       | `warmup_epochs`   | Epoch khởi động LR         |
|                       | `warmup_momentum` | Momentum trong warmup      |
|                       | `warmup_bias_lr`  | Bias LR trong warmup       |
| **Loss**              | `box`             | Hệ số loss cho bbox        |
|                       | `cls`             | Hệ số loss cho class       |
|                       | `obj`             | Hệ số loss cho objectness  |
|                       | `fl_gamma`        | Gamma trong focal loss     |
|                       | `label_smoothing` | Làm mềm nhãn               |
| **Data Augmentation** | `hsv_h`           | Thay đổi hue               |
|                       | `hsv_s`           | Thay đổi saturation        |
|                       | `hsv_v`           | Thay đổi value             |
|                       | `degrees`         | Góc xoay ảnh               |
|                       | `translate`       | Dịch chuyển ảnh            |
|                       | `scale`           | Thay đổi tỉ lệ             |
|                       | `shear`           | Biến dạng shear            |
|                       | `perspective`     | Biến dạng phối cảnh        |
|                       | `flipud`          | Lật dọc                    |
|                       | `fliplr`          | Lật ngang                  |
|                       | `mosaic`          | Xác suất dùng mosaic       |
|                       | `mixup`           | Xác suất dùng mixup        |
|                       | `copy_paste`      | Copy-paste augmentation    |
| **Khác**              | `cache`           | Cache dataset (RAM/disk)   |
|                       | `patience`        | Early stopping             |
|                       | `augment`         | Bật/tắt augmentation       |

## Bước 4: Đánh giá model

### Ý nghĩa của các thông số

#### **box_loss - Bounding Box Loss**

- Đo mức độ chính xác giữa tọa độ hộp giới hạn mô hình dự đoán so với hộp giới hạn được đánh nhãn (Dùng CIoU, DIoU, GIoU)
- Giá trị này càng thấp thì càng tốt

#### **cls_loss - Classification Loss**

- Đo mức độ chính xác giữa nhãn của mô hình dự đoán so với nhãn được đánh nhãn (Dùng BCE - binary cross-entropy)
- Giá trị này càng thấp thì càng tốt

#### **dfl_loss - Distribution Focal Loss**

- Dùng trong các phiên bản YOLO mới như YOLOv8 để cải thiện chất lượng định vị hộp thông qua kỹ thuật phân phối xác suất.
- DFL không chỉ dự đoán tọa độ mà còn học phân phối xác suất cho mỗi tọa độ, giúp định vị mượt và chính xác hơn.
- Giá trị này càng thấp càng tốt

#### **Precision - Độ chính xác**

- Trả lời cho câu hỏi: Trong số tất cả các dự đoán có vật thể, có bao nhiêu là đúng?
- Chỉ số này càng cao càng tốt

$$
Precision = \frac{TP}{TP + FP}
$$

Với:

- $TP$ (True Positive): Dự đoán đúng vật thể
- $FP$ (False Possitive): Dự đoán nhầm, tức mô hình dự đoán có vật thể trong ảnh, nhưng thực tế thì không có

#### **Recall - Độ bao phủ**

- Trả lời cho câu hỏi: Trong số tất cả các vật thể thực sự có trong ảnh, mô hình dự đoán đúng được bao nhiêu?
- Chỉ số này càng cao càng tốt
  $$
  Recall = \frac{TP}{TP + FN}
  $$
  Với:
  - $TP$ (True Positive): Dự đoán đúng vật thể
  - $FN$ (False Negative): Mô hình dự đoán không có vật thể có trong ảnh, nhưng thực tế là có

#### **mAP@0.5 - Mean Average Precision (IoU 0.5)**

- IoU (Intersection over Union): Tỉ lệ chồng lấp giữa box dự đoán và box thực tế.
- mAP@0.5 nghĩa là chỉ cần IoU ≥ 0.5 (50% chồng lấp) là chấp nhận là đúng.
- mAP (mean average precision): trung bình của average precision qua các lớp và các ngưỡng confidence.
- Chỉ số này càng cao tức mô hình dự đoán vị trí của các box càng đúng

#### **mAP@0.5:0.95 – Trung bình AP từ IoU 0.5 đến 0.95**

- Tính mAP ở nhiều mức IoU: 0.5, 0.55, 0.6, ..., 0.95 (tăng mỗi 0.05).
- Sau đó lấy trung bình.

Bảng metric so sánh model, phân tích lỗi.

## Bước 5: Demo

Trong Project, file `demo.py` chứa chương trình Demo.

## Bước 6: Triển khai trên RKNN3955
