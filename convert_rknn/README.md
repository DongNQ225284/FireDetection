# 🚀 Chuyển đổi YOLO sang RKNN (Ubuntu 22.04, RK3588)

Hướng dẫn này giúp bạn chuyển đổi mô hình YOLO từ định dạng `.pt` sang `.onnx` và cuối cùng sang `.rknn` để chạy trên các thiết bị Rockchip (ví dụ: **RK3588**).

## 📑 Mục lục

- [📋 Yêu cầu](#-yêu-cầu)
- [🛠️ B1. Chuẩn bị](#️-b1-chuẩn-bị)
  - [1. Cài đặt Miniconda](#1-cài-đặt-miniconda)
  - [2. Tải xuống các repo liên quan đến RKNN](#2-tải-xuống-các-repo-liên-quan-đến-rknn)
- [🔄 B2. Chuyển đổi pt → onnx](#-b2-chuyển-đổi-pt--onnx)
  - [1. Tạo môi trường ảo](#1-tạo-môi-trường-ảo)
  - [2. Cài đặt package](#2-cài-đặt-package)
  - [3. Export mô hình sang ONNX](#3-export-mô-hình-sang-onnx)
- [🔁 B3. Chuyển đổi onnx → rknn](#-b3-chuyển-đổi-onnx--rknn)
  - [1. Tạo môi trường ảo](#1-tạo-môi-trường-ảo-1)
  - [2. Cài đặt RKNN Toolkit2](#2-cài-đặt-rknn-toolkit2)
  - [3. Export mô hình sang RKNN](#3-thực-hiện-chuyển-đổi-onnx--rknn)

## 📋 Yêu cầu

- Ubuntu 22.04
- Dự án đã có sẵn file `.pt` (mô hình huấn luyện xong từ YOLO)

## 🛠️ B1. Chuẩn bị

### 1. Cài đặt Miniconda

```bash
# Tải Miniconda (Linux)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Chạy cài đặt
bash Miniconda3-latest-Linux-x86_64.sh

# Thêm Miniconda vào biến môi trường
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Kiểm tra cài đặt
conda --version
> conda 25.x.x # ✅ cài thành công
```

### 2. Tải xuống các repo liên quan đến RKNN

```bash
# Tạo folder 'convert'
mkdir convert && cd convert

# Tải repo RKNN optimization for exporting YOLO
git clone https://github.com/airockchip/ultralytics_yolo11.git --depth 1

# Tải repo RKNN-Toolkit2
git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1

# Tải repo RKNN Model Zoo
git clone https://github.com/airockchip/rknn_model_zoo.git --depth 1
```

💡 **Lưu ý:**

- `--depth 1` = chỉ tải phiên bản mới nhất.
- Nếu `git clone` lỗi, có thể tải file `.zip` từ GitHub và giải nén thủ công.

📂 Kiến trúc thư mục sau khi tải về:

```bash
convert
├── rknn-toolkit2
├── rknn_model_zoo
└── ultralytics_yolo11
```

---

## 🔄 B2. Chuyển đổi pt → onnx

### 1. Tạo môi trường ảo

```bash
conda create -n onnx python=3.8 -y
conda activate onnx
```

### 2. Cài đặt package

```bash
cd ultralytics_yolo11

# Cài YOLO + gói export
pip install ultralytics
pip install -e .[export]
```

### 3. Export mô hình sang ONNX

Truy cập tới `./ultralytics/cfg/default.yaml`

```yaml
....
# Train settings -----------------------------------------------
model: <path to file .pt>
.....
```

Sau đó chạy

```bash
python ./ultralytics/engine/exporter.py
# Kết quả: file .onnx sẽ được lưu vào cùng folder với file .pt
```

⚠️ **Lưu ý:** Nếu chạy `pip` sai môi trường → hãy kiểm tra bằng `which pip`.

```bash
which pip
> .../miniconda3/envs/onnx/bin/pip
```

Nếu không khớp, hãy dùng:

```bash
python -m pip install ...
```

## 🔁 B3. Chuyển đổi onnx → rknn

### 1. Tạo môi trường ảo

```bash
conda create -n rknn python=3.8 -y
conda activate rknn
```

### 2. Cài đặt RKNN Toolkit2

```bash
# Cài từ PyPI
pip install rknn-toolkit2 -i https://pypi.org/simple

# Chuyển vào thư mục rknn-toolkit2
cd ../rknn-toolkit2/rknn-toolkit2

# Chọn tệp yêu cầu phù hợp dựa trên phiên bản Python và kiến ​​trúc bộ xử lý:
# trong đó cpxx là phiên bản Python
pip install -r packages/x86_64/requirements_cpxx.txt
# pip install -r packages/arm64/arm64_requirements_cpxx.txt

# Cài đặt RKNN-Toolkit2
# Chọn tệp gói wheel phù hợp dựa trên phiên bản Python và kiến ​​trúc bộ xử lý:
# trong đó x.x.x là số phiên bản RKNN-Toolkit2, cpxx là phiên bản Python
pip install packages/x86_64/rknn_toolkit2-x.x.x-cpxx-cpxx-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# pip install packages/arm64/rknn_toolkit2-x.x.x-cpxx-cpxx-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

💡 **Tips kiểm tra:**

Ví dụ:

```bash
# Kiểm tra kiến trúc
uname -m
> x86_64

# Kiểm tra phiên bản Python
python -V
> Python 3.8.20

# Cài đặt các gói gần thiết (xem tên file trong packages/x86_64)
pip install -r packages/x86_64/requirements_cp38-2.3.2.txt

# Cài đặt RKNN-Toolkit2 (xem tên file trong packages/x86_64)
pip install packages/x86_64/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### 3. Export mô hình sang RKNN

```bash
cd ../../rknn_model_zoo/examples/yolo11/python

python convert.py <onnx_model> <TARGET_PLATFORM> <dtype(optional)> <output_rknn_path(optional)>

# Ví dụ:
python convert.py \
  /mnt/d/Arrow/ProjectCV/fire_detection_rknn/demo/model/yolo11n.onnx \
  rk3588 \
  i8 \
  /mnt/d/Arrow/ProjectCV/fire_detection_rknn/demo/model/yolo11n.rknn
```

📌 **Tham số:**

- `<onnx_model>`: Đường dẫn file `.onnx`
- `<TARGET_PLATFORM>`: Nền tảng NPU, ví dụ `rk3588`
- `<dtype>` _(tùy chọn)_: `i8`, `u8`, `fp`

  - `i8/u8`: lượng tử hóa (cần dataset mẫu để calibrate)
  - `fp`: không lượng tử hóa

- `<output_rknn_path>` _(tùy chọn)_: đường dẫn lưu file `.rknn`

---
