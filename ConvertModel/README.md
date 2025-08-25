# File `.pt`, `.onnx`, `.rknn` là gì?

## 1. File `.pt`

**Nguồn gốc:**

- Định dạng lưu mô hình của PyTorch. Là file thu được sau quá trình huấn luyện

**File `.pt` chứa:**

- Trọng số mô hình (weights) đã huấn luyện.
- Có thể chứa cấu trúc mô hình (nếu dùng `torch.save(model)`).
- Hoặc chỉ chứa state_dict (nếu dùng `torch.save(model.state_dict())`).

**Ưu điểm:**

- Giữ nguyên tất cả tính năng của PyTorch.
- Phù hợp cho nghiên cứu, huấn luyện, fine-tuning.

**Nhược điểm:**

- Chỉ chạy được khi có môi trường PyTorch (khó triển khai trên thiết bị nhúng nếu không cài PyTorch).

## 2. File `.onnx`

**Nguồn gốc:**

- Định dạng Open Neural Network Exchange (ONNX) — chuẩn mở cho mô hình AI.

**Ý nghĩa:**

- Giúp mô hình huấn luyện bằng framework này (PyTorch, TensorFlow, MXNet, …) có thể triển khai trên framework khác.
- Chứa cấu trúc mô hình + trọng số dưới dạng chuẩn trung gian.

**Ưu điểm:**

- Đa nền tảng, tương thích nhiều môi trường.
- Thường tối ưu tốt cho inference (dự đoán).

**Nhược điểm:**

- Không hỗ trợ 100% các layer tùy chỉnh đặc biệt.
- Có thể mất một số thông tin huấn luyện nâng cao.

## 3. File `.rknn`

**Nguồn gốc:**

- Định dạng mô hình dành cho Rockchip NPU (dòng chip AI như RK3399Pro, RK3566, RK3588…).

**Ý nghĩa:**

- Được tạo bằng RKNN Toolkit (do Rockchip cung cấp).
- File `.rknn` là mô hình đã tối ưu hóa để chạy trên NPU của Rockchip.

**Ưu điểm:**

- Tối ưu phần cứng, tốc độ inference rất nhanh.
- Tiết kiệm năng lượng cho thiết bị nhúng.

**Nhược điểm:**

- Chỉ chạy trên chip Rockchip có NPU.
- Quá trình convert cần tuân thủ định dạng hỗ trợ.

| Định dạng | Dùng cho     | Nguồn gốc                    | Khả năng tương thích | Mục tiêu chính                   |
| --------- | ------------ | ---------------------------- | -------------------- | -------------------------------- |
| `.pt`     | PyTorch      | Facebook (Meta)              | Chỉ PyTorch          | Huấn luyện, fine-tune            |
| `.onnx`   | Đa nền tảng  | ONNX (Microsoft & cộng đồng) | Rất cao              | Triển khai cross-framework       |
| `.rknn`   | Rockchip NPU | Rockchip                     | Chỉ Rockchip NPU     | Tối ưu tốc độ cho thiết bị nhúng |

# Quy trình chuyển đổi

**`.pt` -> `.onnx` -> `.rknn`**

### Chuyển đổi từ `.pt` -> `.onnx`

**Chạy file `cvt_onnx.py`**

- Cần tải các thư viện cần thiết

### Chuyển đổi từ `.onnx` -> `.rknn`

#### **1. Tạo dataset: Chạy file `create_dataset.py`**

- Chú ý đường dẫn

#### **2. Chạy file `cvt_rknn.py`**

- Tính đến tháng 8 năm 2025 `rknn.api` **không hỗ trợ Windows**, nên cần chuyển sang Linux để chạy
- Sử dụng `Ubuntu-22.4` thông qua `WSL`
- Cần tải `WSL` và `Ubuntu-22.4`
- Sau đó cần tải thêm các thư viện hỗ trợ

**Sử dụng Ubuntu thông qua WSL, ở giao diện PowerShell**

```bash
wsl -d Ubuntu-22.04
```

**Giao diện Ubuntu, truy cập tới folder file `cvt_rknn.py`**

- Dưới đây là ví dụ:

```bash
cd /mnt/d/ProjectCV/fire_detection/ConvertModel
```

Chạy file `cvt_rknn.py`

```bash
python3 cvt_rknn.py
```
