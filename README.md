# Mô tả project

Project này trình bày hướng dẫn sử dụng YOLO để phát hiện đám cháy trong nhà từ **camera** trong thời gian thực

# Quy trình thực hiện

## Bước 1: Chuẩn bị dữ liệu

- Những nhãn xuất hiện là gì?
- Nguồn ảnh cần như thế nào?

### Xây dựng nhãn

**Nhãn được sử dụng trong Project:**

1. `fire` – cháy thật
2. `smoke` – khói
3. `fake-fire` – cháy giả (mô hình, video, LED giả lửa)
4. `light` – bóng đèn
5. `bright-color` – vùng màu sáng dễ nhầm lẫn
6. `sun` – mặt trời

> Lửa có thể bị nhầm lẫn sang mặt trời, bóng đèn, đám cháy mô hình, vùng màu sáng

### Thu thập dữ liệu

#### Chất lượng dữ liệu

- Độ phân dải cao
- Hình ảnh rõ ràng, ít nhiễu hoặc mờ quá mức

#### Sự đa dạng về dữ liệu

- Đa dạng về môi trường: ngoài trời, trong nhà, nhiều loại cảnh
- Đa dạng về điều kiện ánh sáng: ban ngày, ban đêm, ánh sáng yếu, ánh sáng mạnh.
- Đa dạng về hình dạng và kích thước đối tượng: cháy nhỏ, cháy lớn, cháy bị che khuất, khói, đám lửa lan rộng.
- Đa dạng về background: tránh model học nhầm pattern của nền thay vì lửa.

#### Cân bằng dữ liệu

- Số lượng ảnh có cháy và không cháy tương đối cân bằng.

- Nếu **imbalance**, cần xử lý bằng **augmentation** hoặc **weighting loss**.

### Nguồn dữ liệu được sử dụng

1. `fire`: 2,000 images (khoảng 100–200 video clips ngắn có fire)
2. `smoke`: 1,500 images (50–100 clips)
3. `fake-fire`: 1,000 images
4. `light`: 1,000 images
5. `bright-color`: 800 images
6. `sun`: 500 images
7. **Negatives** (no fire/smoke but similar scenes): 3,000–5,000 images (phong phú ở ban đêm/ban ngày)

| STT | Dataset                                                                                                  |
| --- | -------------------------------------------------------------------------------------------------------- |
| 1   | [Kaggle: FireNet Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)                      |
| 2   | [GitHub: DFire](https://github.com/firenet2021/dataset)                                                  |
| 3   | [BoWFire Dataset](https://bitbucket.org/gbdi/bowfire-dataset/src/master/)                                |
| 4   | [VFire Dataset - IEEE Dataport](https://ieee-dataport.org/open-access/vfire-dataset)                     |
| 5   | [YouTube Fire Dataset List](https://github.com/OlafenwaMoses/FireNET#datasets)                           |
| 6   | [Kaggle: Fire and Smoke Dataset](https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset) |

### Đánh nhãn

- Sử dụng [Roboflow](https://universe.roboflow.com/nguyen-dong-ys7mf/datafire-tnjle) để đánh nhãn, xử lý dữ liệu

- Đánh nhãn chính xác: **bounding box** cần bao trọn đối tượng

## Bước 2: Kiến trúc model

Liệt kê model đã thử (YOLOv5s, YOLOv5m, YOLOv8n…) và lý do chọn.

## Bước 3: Huấn luyện

Đầu vào là tập dữ liệu được đánh nhãn, cấu trúc dữ liệu chuẩn YOLO

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

## Bước 4: Đánh giá model

Bảng metric so sánh model, phân tích lỗi.

## Bước 5: Inference / Demo

Chạy camera live, video test, hoặc lưu detection output.
