"""Per-layer quantization sensitivity (ONNX, normalised-box model so the head is a fair test).

For each YOLOv5 layer 0-24: quantize ONLY that layer to INT8 (everything else stays FP32), score on the first 1000
val images, and report accuracy lost vs the FP32 model on the same images. Big loss = a layer that should stay FP32.
Calibration ranges are cached, so builds are fast; evaluation dominates. Run when the machine is not busy.

    python scripts/layer_sensitivity.py
Output: results/sensitivity.json
"""
import json
import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_onnx_variants as B  # noqa: E402
from common import DAY2, evaluate  # noqa: E402

N_EVAL = 1000
KIND = {0: "Conv", 1: "Conv", 2: "C3", 3: "Conv", 4: "C3 (-> neck)", 5: "Conv", 6: "C3 (-> neck)", 7: "Conv", 8: "C3", 9: "SPPF (-> neck)",
        10: "Conv", 11: "Upsample", 12: "Concat", 13: "C3", 14: "Conv", 15: "Upsample", 16: "Concat", 17: "C3 (P3 out)",
        18: "Conv", 19: "Concat", 20: "C3 (P4 out)", 21: "Conv", 22: "Concat", 23: "C3 (P5 out)", 24: "Detect head"}
PART = lambda i: "backbone" if i < 10 else "neck" if i < 24 else "head"

out_path = DAY2 / "results" / "sensitivity.json"
res = json.loads(out_path.read_text()) if out_path.exists() else {}
B.install_chunked_calibration()
B.install_calibration_cache(500, CalibrationMethod.Percentile, B.SRC_NORM.stem)
work = B.OUT / "_sensitivity"


def score(path, name):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return evaluate(lambda x: s.run(None, {"images": x})[0][0], name, N_EVAL, out_dir=DAY2 / "results" / "dets_sensitivity",
                    verbose=False, box_scale=640.0)


if "fp32" not in res:
    res["fp32"] = score(B.SRC_NORM, "sens_fp32")["mAP50-95"]
    out_path.write_text(json.dumps(res, indent=2))
print("FP32 reference mAP50-95 (first 1000 imgs):", round(res["fp32"], 4), flush=True)

for g in range(25):
    key = f"L{g:02d}"
    if key in res:
        continue
    others = [f"/model.{i}/" for i in range(25) if i != g]
    p = B.build("m1_all_int8", True, 500, tag=f"sens_{key}", exclude_prefixes=others, out_dir=work, src=B.SRC_NORM)
    m = score(p, f"sens_{key}")["mAP50-95"]
    res[key] = m
    out_path.write_text(json.dumps(res, indent=2))
    print(f"layer {g:2d} {KIND[g]:16s} {PART(g):8s} mAP {m:.4f}  lost {(res['fp32'] - m) / res['fp32'] * 100:+.2f}%", flush=True)
    p.unlink()   # keep the folder small
