"""Accuracy of ONNX models run with OpenVINO (INT8 execution can differ slightly from ONNX Runtime).
    python scripts/evaluate_ov.py ov_m0_fp32 ov_r_m3_backbone_neck_int8_pc ...   -> results/accuracy_ov.json
(keys come from scripts/registry.py)"""
import json
import sys
from pathlib import Path

import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAY2, evaluate  # noqa: E402
from registry import jobs  # noqa: E402

out = DAY2 / "results" / "accuracy_ov.json"
res = json.loads(out.read_text()) if out.exists() else {}
want = sys.argv[1:] or [j["key"] for j in jobs(("ov",))]
for j in jobs(("ov",)):
    if j["key"] not in want or j["key"] in res:
        continue
    comp = ov.Core().compile_model(str(j["path"]), "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 4})
    r = evaluate(lambda x: next(iter(comp(x).values()))[0], j["key"], None, verbose=False, box_scale=j["box_scale"])
    r["size_mb"] = (Path(j["path"]).stat().st_size + Path(j["path"]).with_suffix(".bin").stat().st_size) / 1e6      # graph (.xml) + weights (.bin)
    res = json.loads(out.read_text()) if out.exists() else {}
    res[j["key"]] = r
    out.write_text(json.dumps(res, indent=2))
    print(f"{j['key']}: mAP50-95 {r['mAP50-95']:.4f}", flush=True)
