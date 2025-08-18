import os
import random

# Thư mục chứa ảnh train (đường dẫn trong WSL, dùng "/" thay vì "\")
image_dir = "Fire_indoor_data/train/images"
# File output dataset.txt
output_file = "dataset.txt"

# Các đuôi ảnh hợp lệ
valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]

# Số lượng ảnh muốn chọn (None = lấy hết)
num_samples = 100   # đổi thành None nếu muốn lấy tất cả ảnh

# Lấy danh sách ảnh
all_images = []
for root, _, files in os.walk(image_dir):
    for file in files:
        if any(file.lower().endswith(ext) for ext in valid_exts):
            # Chuẩn hóa đường dẫn sang dạng Linux ("/")
            img_path = os.path.join(root, file).replace("\\", "/")
            all_images.append(img_path)

# Nếu số ảnh muốn chọn nhỏ hơn tổng số ảnh -> random
if num_samples is not None and num_samples < len(all_images):
    selected_images = random.sample(all_images, num_samples)
else:
    selected_images = all_images

# Ghi ra dataset.txt
with open(output_file, "w") as f:
    for img_path in selected_images:
        f.write(img_path + "\n")

print(f"✅ Đã tạo {output_file} với {len(selected_images)} ảnh")
