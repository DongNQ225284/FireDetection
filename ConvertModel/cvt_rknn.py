import os
from rknn.api import RKNN  # type: ignore

def cvt_rknn(onnx_path: str, folder_path: str, target_platform: str = "rk3566"):
    # Tạo thư mục output nếu chưa có
    os.makedirs(folder_path, exist_ok=True)

    # Lấy tên file .onnx và đổi sang .rknn
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    rknn_path = os.path.join(folder_path, f"{model_name}.rknn")

    # Tạo RKNN object
    rknn = RKNN()

    # Cấu hình
    rknn.config(
        mean_values=[[0, 0, 0]],   # chỉnh nếu cần preprocessing khác
        std_values=[[1, 1, 1]],
        target_platform=target_platform
    )

    # Load ONNX
    print("--> Loading ONNX model")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=["images"],
        input_size_list=[[1, 3, 416, 416]]   # chỉnh theo input của bạn
    )
    if ret != 0:
        print("Load ONNX model failed!")
        exit(ret)

    # Build
    print("--> Building RKNN model")
    ret = rknn.build(do_quantization=False)  # nếu muốn INT8 thì set True + dataset
    if ret != 0:
        print("Build RKNN model failed!")
        exit(ret)

    # Export
    print("--> Exporting RKNN model")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("Export RKNN model failed!")
        exit(ret)

    print(f"RKNN model exported successfully: {rknn_path}")
    return rknn_path

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(BASE_DIR, "Model", "best_v2_2.onnx")
    folder_path = os.path.join(BASE_DIR, "Model")
    target_platform = "rk3566"
    cvt_rknn(onnx_path, folder_path, target_platform)
