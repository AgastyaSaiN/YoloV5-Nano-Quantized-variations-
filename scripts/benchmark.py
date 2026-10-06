"""Benchmark FP32 + INT8 variants: mAP (COCO val2017 1000-img held-out subset), latency, size."""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"fp32_reference": ROOT / "models" / "original" / "yolov5nu.onnx"}
MODELS.update({p.stem: p for p in sorted((ROOT / "models" / "int8").glob("*.onnx")) if not p.stem.startswith("_")})
results = {}
out_json = ROOT / "results" / "benchmark_results.json"
if out_json.exists():
    results = json.loads(out_json.read_text())

so = ort.SessionOptions()
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

for name, path in MODELS.items():
    if name in results:
        continue
    print(f"== {name}", flush=True)
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    for _ in range(5):
        sess.run(None, {"images": dummy})
    ts = []
    for _ in range(50):
        t = time.perf_counter(); sess.run(None, {"images": dummy}); ts.append((time.perf_counter() - t) * 1000)
    m = YOLO(str(path), task="detect").val(data=str(ROOT / "data" / "coco" / "eval.yaml"), imgsz=640, batch=1,
                                           conf=0.001, iou=0.7, plots=False, verbose=False, device="cpu")
    results[name] = {"mAP50-95": float(m.box.map), "mAP50": float(m.box.map50),
                     "precision": float(m.box.mp), "recall": float(m.box.mr),
                     "latency_ms_median": float(np.median(ts)), "size_mb": path.stat().st_size / 1e6}
    out_json.write_text(json.dumps(results, indent=2))
    print(results[name], flush=True)
