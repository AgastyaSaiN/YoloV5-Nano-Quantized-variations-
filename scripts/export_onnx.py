"""Export YOLOv5n to ONNX at a chosen input size, with the box output normalised to 0-1 (see make_normalized_model.py for why).

    python scripts/export_onnx.py --imgsz 512 --tag f512
Writes models/fp32_512/<tag>.pt, <tag>.onnx (pixel boxes) and <tag>_norm.onnx (boxes 0-1; multiply by the input size after inference).
The weights are not changed: the same trained YOLOv5n (models/fp32/yolov5n.pt), only the input size of the exported graph differs.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper as nh

DAY2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAY2 / "vendor" / "yolov5"))
SCALE_CONSTANTS = ["Constant_3", "Constant_11", "Constant_19", "Constant_6", "Constant_14", "Constant_22"]


def normalise_boxes(src, dst, size):
    """Divide the Detect decode constants (strides and anchor grids) by the input size, so boxes come out as fractions of the picture."""
    m = onnx.load(str(src))
    done = 0
    for node in m.graph.node:
        if node.op_type == "Constant" and node.name.startswith("/model.24/") and node.name.split("/")[-1] in SCALE_CONSTANTS:
            t = nh.to_array(node.attribute[0].t)
            node.attribute[0].t.CopyFrom(nh.from_array((t / float(size)).astype(np.float32), node.attribute[0].t.name))
            done += 1
    assert done == 6, done
    onnx.save(m, str(dst))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DAY2 / "models/fp32/yolov5n.pt"))
    ap.add_argument("--imgsz", type=int, default=512, help="multiple of 32")
    ap.add_argument("--tag", default="f512")
    ap.add_argument("--out", default=str(DAY2 / "models" / "fp32_512"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    pt = out / f"{a.tag}.pt"
    torch.save({"model": model.half(), "ema": None, "epoch": -1}, pt)
    subprocess.run([sys.executable, "export.py", "--weights", str(pt), "--include", "onnx", "--opset", "13", "--imgsz", str(a.imgsz), "--device", "cpu"],
                   cwd=DAY2 / "vendor" / "yolov5", check=True, capture_output=True)
    normalise_boxes(out / f"{a.tag}.onnx", out / f"{a.tag}_norm.onnx", a.imgsz)
    print("wrote", pt.name, f"{a.tag}.onnx", f"{a.tag}_norm.onnx")
