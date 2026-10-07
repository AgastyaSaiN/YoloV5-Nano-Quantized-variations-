"""Shared code for Day 2: data access, YOLOv5 pre/post-processing, COCO scoring (P/R/F1 are computed in compute_prf.py).

Pure numpy + cv2 + pycocotools, so it runs in both the main environment and the TensorFlow .venv.
Pre/post-processing mirrors the official yolov5 val.py (square 640 letterbox, conf 0.001, IoU 0.6,
multi-label NMS, max 300 detections).
"""
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

DAY2 = Path(__file__).resolve().parent.parent
VAL_ZIP = DAY2.parent / "Day 1" / "data" / "coco" / "val2017.zip"       # read-only reuse of Day 1's download
ANN = DAY2 / "data" / "annotations" / "instances_val2017.json"
CALIB_DIR = DAY2 / "data" / "calib_train2017"
IMGSZ = 640
CONF_THRES, IOU_THRES, MAX_DET = 0.001, 0.6, 300


# ----------------------------------------------------------------------------- data
def val_image_ids():
    return sorted(json.loads(ANN.read_text())["images"], key=lambda d: d["id"])


class ValImages:
    """Reads val2017 JPEGs straight out of the zip (no 780 MB copy)."""

    def __init__(self):
        self.zip = zipfile.ZipFile(VAL_ZIP)
        self.images = val_image_ids()

    def __len__(self):
        return len(self.images)

    def read(self, i):
        info = self.images[i]
        buf = np.frombuffer(self.zip.read(f"val2017/{info['file_name']}"), np.uint8)
        return info["id"], cv2.imdecode(buf, cv2.IMREAD_COLOR)


def coco_category_ids():
    cats = sorted(c["id"] for c in json.loads(ANN.read_text())["categories"])
    assert len(cats) == 80
    return cats


# ----------------------------------------------------------------------------- pre/post
def letterbox(img, size=IMGSZ, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, (left, top)


def preprocess(img_bgr, size=IMGSZ):
    """BGR image -> (1,3,size,size) float32 in [0,1], plus (ratio, pad) to undo the letterbox."""
    img, r, pad = letterbox(img_bgr, size)
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    return x, r, pad


def nms_numpy(boxes, scores, iou_thres):
    """Greedy NMS on xyxy boxes already offset per class. Returns kept indices (score-descending)."""
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep, idx = [], np.arange(len(order))
    while idx.size and len(keep) < MAX_DET:
        i = idx[0]
        keep.append(i)
        rest = idx[1:]
        if not rest.size:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        idx = rest[iou <= iou_thres]
    return order[np.array(keep, dtype=int)]


def postprocess(pred, r, pad, orig_shape, box_scale=1.0):
    """pred: (25200, 85) = xywh(pixels), obj, 80 class scores. Returns (N,6) x1,y1,x2,y2,conf,cls in original image coords."""
    pred = pred.astype(np.float32)
    scores = pred[:, 5:] * pred[:, 4:5]                       # class conf = obj * cls
    i, j = np.nonzero(scores > CONF_THRES)                    # multi-label, as in official val
    if not i.size:
        return np.zeros((0, 6), np.float32)
    xywh = pred[i, :4] * box_scale                              # box_scale=640 for models with 0-1 box output
    box = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                    xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    conf = scores[i, j]
    if len(conf) > 30000:                                     # max_nms
        top = conf.argsort()[::-1][:30000]
        box, conf, j = box[top], conf[top], j[top]
    keep = nms_numpy(box + j[:, None].astype(np.float32) * 7680, conf, IOU_THRES)
    out = np.concatenate([box[keep], conf[keep, None], j[keep, None].astype(np.float32)], 1)
    out[:, [0, 2]] = ((out[:, [0, 2]] - pad[0]) / r).clip(0, orig_shape[1])
    out[:, [1, 3]] = ((out[:, [1, 3]] - pad[1]) / r).clip(0, orig_shape[0])
    return out


# ----------------------------------------------------------------------------- evaluation driver
def evaluate(predict, name, n_images=None, out_dir=None, verbose=True, box_scale=1.0):
    """predict(x: (1,3,640,640) float32) -> (25200,85) float32 raw detections (xywh pixels, obj, cls).

    Runs the full COCO val2017 set (or the first n_images), saves detections, returns a metrics dict.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    out_dir = Path(out_dir or DAY2 / "results" / "dets")
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = coco_category_ids()
    coco = COCO(str(ANN))
    data = ValImages()
    n = len(data) if n_images is None else min(n_images, len(data))
    ids, results = [], []
    t_pre, t_inf, t_post = [], [], []
    for i in range(n):
        img_id, img = data.read(i)
        t0 = time.perf_counter(); x, r, pad = preprocess(img)
        t1 = time.perf_counter(); pred = predict(x)
        t2 = time.perf_counter(); det = postprocess(np.asarray(pred).reshape(-1, 85), r, pad, img.shape[:2], box_scale)
        t3 = time.perf_counter()
        t_pre.append(t1 - t0); t_inf.append(t2 - t1); t_post.append(t3 - t2)
        ids.append(img_id)
        for d in det:
            x1, y1, x2, y2, c, k = d
            results.append({"image_id": img_id, "category_id": cats[int(k)],
                            "bbox": [round(float(x1), 3), round(float(y1), 3), round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                            "score": round(float(c), 5)})
        if verbose and (i + 1) % 500 == 0:
            print(f"  {name}: {i + 1}/{n}", flush=True)

    (out_dir / f"{name}.json").write_text(json.dumps(results))
    if results:
        E = COCOeval(coco, coco.loadRes(str(out_dir / f"{name}.json")), "bbox")
        E.params.imgIds = ids
        E.evaluate(); E.accumulate(); E.summarize()
        s = E.stats
    else:
        s = [0.0] * 12
    ms = lambda v: float(np.median(v) * 1000)
    return {"name": name, "n_images": n,
            "mAP50-95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
            "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
            "AR100": float(s[8]),
            "ms_pre": ms(t_pre), "ms_infer_inrun": ms(t_inf), "ms_post": ms(t_post)}
