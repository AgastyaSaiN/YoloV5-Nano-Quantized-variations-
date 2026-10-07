"""Split the (normalised) FP32 YOLOv5n ONNX at the backbone / neck / head boundaries, and verify the chain.

Boundary tensors (YOLOv5 v7.0):
  backbone -> neck : outputs of layers 4, 6, 9
  neck     -> head : outputs of layers 17, 20, 23

Parts written to models/tflite/_work/onnx_parts/:
  backbone       images        -> [b4, b6, b9]
  neck_head      [b4, b6, b9]  -> output0        (M2: backbone INT8 | neck+head FP32)
  backbone_neck  images        -> [n17, n20, n23]
  head           [n17,n20,n23] -> output0        (M3: backbone+neck INT8 | head FP32)
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import extract_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAY2, ValImages, preprocess  # noqa: E402

SRC = DAY2 / "models" / "fp32" / "yolov5n_norm.onnx"
OUT = DAY2 / "models" / "tflite" / "_work" / "onnx_parts"
B = ["/model.4/cv3/act/Mul_output_0", "/model.6/cv3/act/Mul_output_0", "/model.9/cv2/act/Mul_output_0"]
N = ["/model.17/cv3/act/Mul_output_0", "/model.20/cv3/act/Mul_output_0", "/model.23/cv3/act/Mul_output_0"]
PARTS = {
    "backbone": (["images"], B),
    "neck_head": (B, ["output0"]),
    "backbone_neck": (["images"], N),
    "head": (N, ["output0"]),
}

OUT.mkdir(parents=True, exist_ok=True)
for name, (ins, outs) in PARTS.items():
    extract_model(str(SRC), str(OUT / f"{name}.onnx"), ins, outs)
    m = onnx.load(str(OUT / f"{name}.onnx"))
    print(f"{name:14s} nodes={len(m.graph.node):4d} inputs={[i.name.split('/')[1] if '/' in i.name else i.name for i in m.graph.input]}")

# verify: chained parts == full model
sess = {n: ort.InferenceSession(str(OUT / f"{n}.onnx"), providers=["CPUExecutionProvider"]) for n in PARTS}
full = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"])


def run(name, feeds):
    s = sess[name]
    return s.run(None, dict(zip([i.name for i in s.get_inputs()], feeds)))


data = ValImages()
worst = 0.0
for i in range(3):
    x = preprocess(data.read(i)[1])[0]
    ref = full.run(None, {"images": x})[0]
    a = run("neck_head", run("backbone", [x]))[0]
    b = run("head", run("backbone_neck", [x]))[0]
    worst = max(worst, float(np.abs(ref - a).max()), float(np.abs(ref - b).max()))
print(f"max |full - chained parts| = {worst:.2e}")
