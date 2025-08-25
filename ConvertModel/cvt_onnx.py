import torch #type: ignore
import os
from ultralytics import YOLO #type: ignore

class BackboneNeckWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.model[:-1]
    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer.f, int):
                inp = outputs[layer.f] if layer.f != -1 else x
            elif isinstance(layer.f, list):
                inp = [outputs[j] if j != -1 else x for j in layer.f]
            else:
                inp = x
            out = layer(inp)
            outputs.append(out)
            x = out
        return outputs[-3:]

def cvt_onnx(model_path: str, folder_path: str):
    # Load YOLO model
    yolo = YOLO(model_path)
    model = BackboneNeckWrapper(yolo.model)
    model.eval()

    # Chuẩn bị input giả
    dummy_input = torch.randn(1, 3, 416, 416)

    # Tạo tên file .onnx theo tên file .pt
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_path = os.path.join(folder_path, f"{model_name}.onnx")

    # Xuất ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["images"],
        output_names=["P3", "P4", "P5"],
        opset_version=17,
        dynamic_axes={"images": {0: "batch"}}
    )
    print(f"Export thành công: {onnx_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "Model", "best_v2_2.pt")
    folder_path = os.path.join(BASE_DIR, "Model")
    cvt_onnx(model_path, folder_path)

