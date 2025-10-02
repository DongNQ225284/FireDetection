# 🚀 Demo mô hình RKNN trên Rockchip RK3588

Hướng dẫn này giúp bạn triển khai mô hình RKNN trên thiết bị Rockchip RK3588

## 📑 Mục lục

- [📋 Yêu cầu](#📋-yêu-cầu)
- [🔑 Thiết lập SSH Key](#🔑-thiết-lập-ssh-key)
- [🌐 Truy cập vào Server](#🌐-truy-cập-vào-server)
  - [⬆️ Upload file/thư mục từ máy local lên server](#⬆️-upload-filethư-mục-từ-máy-local-lên-server)
- [🖥️ Kết nối SSH bằng VS Code](#🖥️-kết-nối-ssh-bằng-vs-code)
  - [1. Cài đặt Extension](#1-cài-đặt-extension)
  - [2. Thêm cấu hình kết nối](#2-thêm-cấu-hình-kết-nối)
  - [3. File config mẫu](#3-file-config-mẫu)
  - [4. Kết nối](#4-kết-nối)
  - [5. Ưu điểm](#5-ưu-điểm)
- [🔥 Giới thiệu về mô hình](#🔥-giới-thiệu-về-mô-hình)
  - [1. Tạo môi trường ảo](#1-tạo-môi-trường-ảo)
  - [2. Cài đặt RKNN Toolkit2](#2-cài-đặt-rknn-toolkit2)
  - [3. Chạy demo](#3-chạy-demo)

## 📋 Yêu cầu

- Thiết bị **Rockchip RK3588** đã được cài đặt hệ điều hành
- Tài khoản truy cập (user/IP) do quản trị viên cung cấp

## 🔑 Thiết lập SSH Key

1. **Tạo SSH Key** (chạy trên **PowerShell** Windows):

   ```bash
   ssh-keygen -t ed25519 -C "youremail@example.com"
   ```

   - Nhấn **Enter** để lưu mặc định tại `C:\Users\UserName\.ssh\id_ed25519`
   - Nhập passphrase (hoặc bỏ trống nếu không cần)

2. **Xem SSH Public Key**:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

   - Copy toàn bộ nội dung và gửi cho quản trị viên để thêm vào server

💡 **Giải thích ngắn gọn về SSH key**

- SSH key có 2 phần:

  - 🔒 **Private key** (`id_ed25519`) → giữ bí mật, không chia sẻ
  - 🔑 **Public key** (`id_ed25519.pub`) → đưa cho server để xác thực

Khi kết nối, server sẽ kiểm tra public key → giúp đăng nhập an toàn mà **không cần mật khẩu**.

## 🌐 Truy cập vào Server

Giả sử quản trị viên cung cấp thông tin:
`root@192.168.168.24`

Kết nối SSH:

```bash
ssh root@192.168.168.24
```

- Lần đầu sẽ hỏi: `Are you sure you want to continue connecting?` → gõ **yes**
- Nếu có passphrase, nhập passphrase để xác thực
- Sau đó sẽ vào được Terminal của server **RK3588** 🎉

### ⬆️ Upload file/thư mục từ máy local lên server

```bash
scp -r "<path_in_your_pc>" username@IP:<path_folder_in_server>
```

- `-r`: copy cả thư mục (recursive)
- `<path_in_your_pc>`: đường dẫn file/thư mục trên máy local
- `<path_folder_in_server>`: nơi muốn lưu trên server

**Thực hiện upload, ví dụ:**

```bash
scp -r "D:\Arrow\ProjectCV\fire_detection_rknn" root@192.168.168.24:/root/DongNguyen/
```

## 🖥️ Kết nối SSH bằng VS Code

Ngoài việc dùng terminal, bạn có thể kết nối trực tiếp vào RK3588 bằng **Visual Studio Code** để code và quản lý file thuận tiện hơn.

### 1. Cài đặt Extension

- Mở **VS Code**
- Vào **Extensions Marketplace** (Ctrl+Shift+X)
- Cài đặt plugin: **Remote - SSH**

### 2. Thêm cấu hình kết nối

1. Mở Command Palette (Ctrl+Shift+P) → chọn **Remote-SSH: Add New SSH Host**

2. Nhập thông tin kết nối, ví dụ:

   ```
   ssh root@192.168.168.24
   ```

3. Chọn file config để lưu (`~/.ssh/config`)

### 3. File config mẫu

Mở file `~/.ssh/config` và chỉnh như sau:

```ssh
Host rk3588
    HostName 192.168.168.24
    User root
    IdentityFile ~/.ssh/id_ed25519
```

- `rk3588` → tên bạn đặt để dễ nhớ
- `HostName` → địa chỉ IP của RK3588
- `User` → user do quản trị viên cung cấp (thường là `root`)
- `IdentityFile` → đường dẫn tới private key của bạn

### 4. Kết nối

- Mở Command Palette → chọn **Remote-SSH: Connect to Host…**
- Chọn `rk3588` (theo tên đã đặt trong config)
- VS Code sẽ mở một cửa sổ mới kết nối trực tiếp vào RK3588

### 5. Ưu điểm

- ✍️ Chỉnh sửa file trực tiếp trên RK3588 như trên máy local
- 📂 Quản lý project trong Explorer
- 🐞 Debug code Python hoặc C++ ngay trên thiết bị
- 🔄 Đồng bộ code nhanh hơn so với upload/download thủ công

## 🔥 Giới thiệu về mô hình

Mô hình `.rknn` được sử dụng là mô hình nhận diện lửa, được chuyển đổi từ mô hình `yolo11n.pt`.

Để biết chi tiết cách **chuyển đổi YOLO pt → ONNX → RKNN**, hãy xem tại: [📘 Hướng dẫn Convert](../convert/README.md).

### 1. Tạo môi trường ảo

```bash
# Tạo môi trường demo
conda create -n demo python=3.8

# Kích hoạt môi trường
conda activate demo
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

### 3. Chạy demo

Trong thư mục [`fire_detection_rknn/demo`](demo) có file [`main.py`](demo/main.py) là script demo nhận diện lửa với đầu vào là ảnh:

```bash
python main.py <path_to_model> <path_to_images> <save_to_path>

# Ví dụ
python main.py model/yolo11n.rknn input output
```

_Trong đó:_

- `<path_to_model>`: đường dẫn đến model nhận diện
- `<path_to_images>`: đường dẫn đến folder ảnh
- `<save_to_path>`: đường dẫn folder lưu kết quả
