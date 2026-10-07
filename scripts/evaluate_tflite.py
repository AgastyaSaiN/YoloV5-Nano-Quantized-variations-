"""Evaluate a TFLite pipeline folder on COCO val2017 (TensorFlow venv).
   .venv\Scripts\python.exe scripts/evaluate_tflite.py models/tflite/m3_backbone_neck_int8_pc [--n 200]
Box output of these models is normalised to 0-1, hence box_scale=640."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAY2, evaluate  # noqa: E402
from tflite_runner import Pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--n", type=int, default=None)
a = ap.parse_args()
folder = Path(a.folder)
name = "tflite_" + folder.name
pipe = Pipeline(folder)
res = evaluate(pipe, name, a.n, box_scale=640.0)
res["size_mb"] = pipe.size_mb
out = DAY2 / "results" / ("accuracy_tflite_quick.json" if a.n else "accuracy_tflite.json")
allr = json.loads(out.read_text()) if out.exists() else {}
allr[name] = res
out.write_text(json.dumps(allr, indent=2))
print(json.dumps({k: res[k] for k in ("mAP50-95", "mAP50", "F1@0.25", "size_mb")}, indent=2))
