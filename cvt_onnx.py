import torch
from ultralytics import YOLO

class BackboneNeckWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # bỏ Detect head cuối
        self.layers = model.model[:-1]

    def forward(self, x):
        outputs = []  # lưu output của mỗi layer
        for i, layer in enumerate(self.layers):
            # nếu layer lấy input từ nhiều node trước (Concat, Add,...)
            if isinstance(layer.f, int):
                inp = outputs[layer.f] if layer.f != -1 else x
            elif isinstance(layer.f, list):
                inp = [outputs[j] if j != -1 else x for j in layer.f]
            else:
                inp = x

            out = layer(inp)
            outputs.append(out)
            x = out  # cập nhật x cho trường hợp f = -1
        # cuối backbone+neck trả ra list feature maps
        return outputs[-3:]  # P3, P4, P5

if __name__ == "__main__":
    yolo = YOLO("best.pt")
    model = BackboneNeckWrapper(yolo.model)
    model.eval()

    dummy_input = torch.randn(1, 3, 416, 416)

    # Export sang ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "best_raw.onnx",
        input_names=["images"],
        output_names=["P3", "P4", "P5"],
        opset_version=17,   # Torch 2.1.2 chỉ hỗ trợ tới 17
        dynamic_axes={"images": {0: "batch"}}
    )

    print("✅ Export thành công best_raw.onnx")
