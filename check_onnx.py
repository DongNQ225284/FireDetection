import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("best_raw.onnx")
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

# Dùng đúng kích thước input
x = np.random.randn(1, 3, 416, 416).astype(np.float32)

outputs = sess.run(output_names, {input_name: x})

for name, out in zip(output_names, outputs):
    print(name, out.shape)
