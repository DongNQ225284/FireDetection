import os
import random

def create_dataset(data_path, output_file="dataset.txt", num_samples=None):
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]

    # Quét toàn bộ ảnh
    all_images = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_exts):
                img_path = os.path.join(root, file).replace("\\", "/")
                all_images.append(img_path)

    if not all_images:
        raise ValueError(f"Không tìm thấy ảnh trong {data_path}")

    # Random chọn ảnh nếu cần
    if num_samples is not None and num_samples < len(all_images):
        selected_images = random.sample(all_images, num_samples)
    else:
        selected_images = all_images

    # Ghi ra file dataset.txt
    with open(output_file, "w") as f:
        for img_path in selected_images:
            f.write(img_path + "\n")

    print(f"Đã tạo {output_file} với {len(selected_images)} ảnh")
    return os.path.abspath(output_file)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "Data", "Fire_indoor_data_v4", "train", "images")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.txt")
    create_dataset(data_path, output_file, num_samples=100)
