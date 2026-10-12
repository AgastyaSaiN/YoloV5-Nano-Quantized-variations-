"""Swap SiLU -> ReLU in the backbone and neck (layers 0-23) of YOLOv5n; the Detect head (layer 24) is not touched.

ReLU fuses into the convolution in INT8 (no separate sigmoid/multiply kernels) and is supported by every NPU.
Without retraining the accuracy is poor; this script is used (a) to measure the SPEED POTENTIAL and (b) as the
starting point for fine-tuning (scripts/finetune_relu.py).

    python scripts/make_relu_model.py [--weights path.pt] [--tag relu_untrained]
Writes models/relu/<tag>.pt, <tag>.onnx (pixel boxes) and <tag>_norm.onnx (boxes 0-1, as in the other models).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper as nh

DAY2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAY2 / "vendor" / "yolov5"))
OUT = DAY2 / "models" / "relu"
SCALE_CONSTANTS = ["Constant_3", "Constant_11", "Constant_19", "Constant_6", "Constant_14", "Constant_22"]


def swap_silu_to_relu(model, act="relu"):
    make = {"relu": lambda: nn.ReLU(inplace=True), "hardswish": lambda: nn.Hardswish(inplace=True), "relu6": lambda: nn.ReLU6(inplace=True)}[act]
    n = 0
    for i, layer in enumerate(model.model):
        if i >= 24:                                     # never touch the Detect head
            continue
        for mod in layer.modules():
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.SiLU):
                    setattr(mod, name, make()); n += 1
    return n


def normalise_boxes(src, dst, size=640):
    m = onnx.load(str(src)); done = 0
    for node in m.graph.node:
        if node.op_type == "Constant" and node.name.startswith("/model.24/") and node.name.split("/")[-1] in SCALE_CONSTANTS:
            t = nh.to_array(node.attribute[0].t)
            node.attribute[0].t.CopyFrom(nh.from_array((t / float(size)).astype(np.float32), node.attribute[0].t.name)); done += 1
    assert done == 6, done
    onnx.save(m, str(dst))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DAY2 / "models/fp32/yolov5n.pt"))
    ap.add_argument("--tag", default="relu_untrained")
    ap.add_argument("--act", default="relu", choices=["relu", "hardswish", "relu6"])
    ap.add_argument("--opset", default="13")
    ap.add_argument("--imgsz", type=int, default=640, help="export size (multiple of 32); boxes are normalised by this size")
    ap.add_argument("--keep-acts", action="store_true", help="export the checkpoint as-is (hybrid models keep SiLU in some layers)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    if not a.keep_acts and any(isinstance(m, nn.SiLU) for m in model.modules()):
        print("swapped", swap_silu_to_relu(model, a.act), f"SiLU -> {a.act} (layers 0-23)")
    pt = OUT / f"{a.tag}.pt"
    torch.save({"model": model.half(), "ema": None, "epoch": -1}, pt)
    subprocess.run([sys.executable, "export.py", "--weights", str(pt), "--include", "onnx", "--opset", a.opset, "--imgsz", str(a.imgsz), "--device", "cpu"],
                   cwd=DAY2 / "vendor" / "yolov5", check=True, capture_output=True)
    normalise_boxes(OUT / f"{a.tag}.onnx", OUT / f"{a.tag}_norm.onnx", a.imgsz)
    print("wrote", pt.name, f"{a.tag}.onnx", f"{a.tag}_norm.onnx")
