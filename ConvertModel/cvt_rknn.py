from rknn.api import RKNN # type: ignore
 
# 1. Tạo RKNN object
rknn = RKNN()

# 1.1 Cấu hình model
rknn.config(
    mean_values=[[0, 0, 0]],  # nếu model đã normalize [-1,1] hoặc 0-1, để [0,0,0]
    std_values=[[1, 1, 1]],   # tương tự
    target_platform='rk3566'  # hoặc rk3588, tùy thiết bị NPU
)

# 2. Load ONNX model
print('--> Loading ONNX model')
ret = rknn.load_onnx(
    model='best_raw.onnx',
    inputs=['images'],         # tên input trong ONNX
    input_size_list=[[1, 3, 416, 416]]  # batch=1
)
if ret != 0:
    print('Load ONNX model failed!')
    exit(ret)

# 3. Build model
print('--> Building RKNN model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build RKNN model failed!')
    exit(ret)

# 4. Export RKNN model
print('--> Exporting RKNN model')
ret = rknn.export_rknn('best.rknn')
if ret != 0:
    print('Export RKNN model failed!')
    exit(ret)

print('RKNN model exported successfully!')
