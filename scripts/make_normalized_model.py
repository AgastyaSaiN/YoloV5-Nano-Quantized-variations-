"""Make a copy of the FP32 YOLOv5n ONNX whose box output (x, y, w, h) is normalised to 0-1 instead of pixels (0-640).

The official yolov5 TFLite export does the same. Why it matters for INT8: the Detect output is one tensor holding
boxes AND confidence scores. With boxes up to 640 and scores below 1 sharing one INT8 scale, every score rounds
to 0 (Day 1's collapsed control). With boxes in 0-1 both fit one scale.

Done by dividing the decode constants (strides for xy, anchor grids for wh) by 640 - no extra graph nodes.
The pipeline multiplies boxes back by 640 after inference (common.postprocess box_scale=640).
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper as nh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAY2, ValImages, preprocess  # noqa: E402

SRC = DAY2 / "models" / "fp32" / "yolov5n.onnx"
DST = DAY2 / "models" / "fp32" / "yolov5n_norm.onnx"
SCALE_CONSTANTS = ["Constant_3", "Constant_11", "Constant_19",     # xy strides (8, 16, 32)
                   "Constant_6", "Constant_14", "Constant_22"]     # wh anchor grids

m = onnx.load(str(SRC))
done = []
for n in m.graph.node:
    if n.op_type == "Constant" and n.name.startswith("/model.24/") and n.name.split("/")[-1] in SCALE_CONSTANTS:
        t = nh.to_array(n.attribute[0].t)
        n.attribute[0].t.CopyFrom(nh.from_array((t / 640.0).astype(np.float32), n.attribute[0].t.name))
        done.append(n.name)
assert len(done) == 6, done
onnx.save(m, str(DST))

a, b = (ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in (SRC, DST))
data = ValImages()
worst = 0
for i in range(5):
    x = preprocess(data.read(i)[1])[0]
    ya, yb = a.run(None, {"images": x})[0], b.run(None, {"images": x})[0]
    yb = yb.copy(); yb[..., :4] *= 640
    worst = max(worst, float(np.abs(ya - yb).max()))
print(f"saved {DST.name}; max |original - normalised*640| over 5 images = {worst:.2e}")
print("normalised box range:", float(b.run(None, {"images": x})[0][..., :4].min()), "to", float(b.run(None, {"images": x})[0][..., :4].max()))
