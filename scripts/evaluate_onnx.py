"""Evaluate one ONNX model on COCO val2017.   python scripts/evaluate_onnx.py models/fp32/yolov5n.onnx [--n 200]"""
import argparse
import json
import sys
from pathlib import Path

import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAY2, evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--n", type=int, default=None, help="only the first N images (quick test)")
ap.add_argument("--name", default=None)
ap.add_argument("--box-scale", type=float, default=1.0, help="640 for the normalised-box model")
a = ap.parse_args()

path = Path(a.model)
name = a.name or path.stem
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
res = evaluate(lambda x: sess.run(None, {inp: x})[0][0], name, a.n, box_scale=a.box_scale)
res["size_mb"] = path.stat().st_size / 1e6
out = DAY2 / "results" / ("accuracy_quick.json" if a.n else "accuracy.json")
out.parent.mkdir(exist_ok=True)
allr = json.loads(out.read_text()) if out.exists() else {}
allr[name] = res
out.write_text(json.dumps(allr, indent=2))
print(json.dumps(res, indent=2))
