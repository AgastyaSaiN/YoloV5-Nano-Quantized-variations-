"""How many calibration images are enough?  M3 (backbone+neck INT8, head FP32), per-channel, calibrated on
N = 50 / 100 / 250 train2017 images (500 is the main result), each scored on all 5000 val images.

    python scripts/calib_size_study.py        -> results/calib_size.json
"""
import json
import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_onnx_variants as B  # noqa: E402
from common import DAY2, evaluate  # noqa: E402

import subprocess

out_path = DAY2 / "results" / "calib_size.json"
res = json.loads(out_path.read_text()) if out_path.exists() else {}
work = B.OUT / "_calib_size"
if len(sys.argv) == 1:       # driver: one fresh process per size (the cache patch must be installed only once per process)
    for n in (50, 100, 250):
        if str(n) not in res:
            subprocess.run([sys.executable, __file__, str(n)], check=True)
    sys.exit()
for n in [int(sys.argv[1])]:
    B.install_chunked_calibration()
    B.install_calibration_cache(n, CalibrationMethod.Percentile, B.SRC.stem)
    p = B.build("m3_backbone_neck_int8", True, n, tag=f"m3_pc_calib{n}", out_dir=work)
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    r = evaluate(lambda x: s.run(None, {"images": x})[0][0], f"calibsize_{n}", None, out_dir=DAY2 / "results" / "dets_calibsize", verbose=False)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res[str(n)] = {k: r[k] for k in ("mAP50-95", "mAP50", "F1@0.25")}
    out_path.write_text(json.dumps(res, indent=2))
    print(f"N={n}: mAP50-95 {r['mAP50-95']:.4f}", flush=True)
    p.unlink()
