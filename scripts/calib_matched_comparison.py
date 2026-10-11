"""Calibration-matched ONNX vs TFLite comparison.

ONNX Runtime variants were calibrated with the percentile method; the TFLite converter can only use min-max. This
rebuilds every ONNX variant with MIN-MAX calibration (same 500 images) and scores it, so each design can be compared
with its TFLite counterpart under the same calibration method.
    python scripts/calib_matched_comparison.py  -> results/calib_matched.json
"""
import json
import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import MinMaxCalibrater

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_onnx_variants as B  # noqa: E402
from common import DAY2, evaluate  # noqa: E402

out = DAY2 / "results" / "calib_matched.json"
work = B.OUT / "_calib_matched"
GROUPS = {"norm": (B.SRC_NORM, 640.0, ["m1_all_int8"]), "pixel": (B.SRC, 1.0, ["m2_backbone_int8", "m3_backbone_neck_int8"])}
if len(sys.argv) == 1:                       # driver: one process per source model (patches must be installed once per process)
    import subprocess
    for grp in GROUPS:
        subprocess.run([sys.executable, __file__, grp], check=True)
    sys.exit()
src, scale, designs = GROUPS[sys.argv[1]]
B.install_chunked_calibration(chunk=20, cls=MinMaxCalibrater)
# no range cache for min-max: ORT's MinMaxCalibrater calls compute_data() after every chunk, so a cache would freeze the ranges after the first 20 images
for design in designs:
    for g in ("pc", "pt"):
        key = f"{design}_{g}"
        res = json.loads(out.read_text()) if out.exists() else {}
        if key in res:
            continue
        p = B.build(design, g == "pc", 500, method=CalibrationMethod.MinMax, tag=f"{key}_minmax", out_dir=work, src=src)
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        r = evaluate(lambda x: s.run(None, {"images": x})[0][0], f"minmax_{key}", None, out_dir=DAY2 / "results" / "dets_calibmatched", verbose=False, box_scale=scale)
        res = json.loads(out.read_text()) if out.exists() else {}
        res[key] = {"mAP50-95": r["mAP50-95"], "mAP50": r["mAP50"]}
        out.write_text(json.dumps(res, indent=2))
        print(f"{key}: min-max ONNX mAP50-95 {r['mAP50-95']:.4f}", flush=True)
        p.unlink()
