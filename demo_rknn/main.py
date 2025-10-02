import os
import cv2
import sys
import argparse
import numpy as np
from py_utils.coco_utils import COCO_test_helper
from py_utils.rknn_executor import RKNN_model_container

# ==== Config ====
OBJ_THRESH = 0.25      # Ngưỡng confidence (xác suất đối tượng). 
                       # Chỉ giữ lại các bounding box có độ tin cậy >= 0.25.
                       # Nếu model phát hiện nhưng độ tin cậy < 0.25 thì sẽ bị loại bỏ.

NMS_THRESH = 0.45      # Ngưỡng IoU cho Non-Maximum Suppression (NMS).
                       # Dùng để loại bỏ các box trùng lặp. 
                       # Nếu hai box cùng class có IoU > 0.45 thì chỉ giữ lại box có score cao hơn.

IMG_SIZE = (640, 640)  # Kích thước ảnh đầu vào cho model (width, height).
                       # Tất cả ảnh đưa vào sẽ được resize + letterbox về 640x640 trước khi infer.

CLASSES = ("fire", "non-fire")  
                       # Danh sách các lớp (classes) mà model có thể phát hiện.
                       # Ở đây model chỉ có 2 lớp: 
                       #  - "fire": vùng có lửa
                       #  - "non-fire": vùng không phải lửa (background hoặc vùng khác)

# ==== Utils ====
def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def dfl(position):
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w).softmax(2)
    acc = torch.arange(mc).reshape(1, 1, mc, 1, 1).float()
    y = (y * acc).sum(2)
    return y.numpy()

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid = np.stack((col, row), axis=0).reshape(1, 2, grid_h, grid_w)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy1 = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy1 * stride, box_xy2 * stride), axis=1)

def post_process(outputs):
    boxes, scores, classes_conf = [], [], []
    branch = 3
    for i in range(branch):
        boxes.append(box_process(outputs[2*i]))
        classes_conf.append(outputs[2*i+1])
        scores.append(np.ones_like(outputs[2*i+1][:, :1, :, :], dtype=np.float32))

    def flatten(x):
        return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])

    boxes = np.concatenate([flatten(b) for b in boxes])
    classes_conf = np.concatenate([flatten(c) for c in classes_conf])
    scores = np.concatenate([flatten(s) for s in scores])

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, s = boxes[inds], scores[inds]
        keep = nms_boxes(b, s)
        if len(keep):
            nboxes.append(b[keep])
            nclasses.append(np.full(len(keep), c))
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

def draw(img, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = f"{CLASSES[cl]} {score:.2f}"
        color = (0, 0, 255) if cl == 0 else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# ==== Main Demo ====
def run_demo(path_to_rknn, path_to_imgs, save_to_path):
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)

    model = RKNN_model_container(path_to_rknn, target="rk3588", device_id=None)
    co_helper = COCO_test_helper(enable_letter_box=True)

    imgs = [f for f in os.listdir(path_to_imgs) if f.lower().endswith((".jpg",".png",".jpeg"))]

    for i, name in enumerate(imgs):
        print(f"[{i+1}/{len(imgs)}] {name}")
        img_src = cv2.imread(os.path.join(path_to_imgs, name))
        if img_src is None: continue

        img_in = co_helper.letter_box(im=img_src.copy(), new_shape=IMG_SIZE, pad_color=(0,0,0))
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

        outputs = model.run([img_in])
        boxes, classes, scores = post_process(outputs)

        result = img_src.copy()
        if boxes is not None:
            draw(result, co_helper.get_real_box(boxes), scores, classes)

        save_path = os.path.join(save_to_path, name)
        cv2.imwrite(save_path, result)
        print("Saved:", save_path)

    model.release()

# ==== Entry point ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RKNN Fire Detection Demo")
    parser.add_argument("model", help="Path to .rknn model")
    parser.add_argument("images", help="Path to images folder")
    parser.add_argument("output", help="Path to save results")
    args = parser.parse_args()

    run_demo(args.model, args.images, args.output)
