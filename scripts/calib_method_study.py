"""Does the calibration method explain why ONNX scores higher than TFLite?
M3 per-channel (ONNX) calibrated with MIN-MAX (the TFLite converter's method) instead of percentile, same 500 images,
scored on all 5000 val images. Compare with the percentile result (main run) and the TFLite result.
    python scripts/calib_method_study.py -> results/calib_method.json"""
import json
import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import MinMaxCalibrater

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_onnx_variants as B  # noqa: E402
from common import DAY2, evaluate  # noqa: E402

work = B.OUT / "_calib_method"
B.install_chunked_calibration(chunk=20, cls=MinMaxCalibrater)   # ORT 1.22's own memory cap discards data; feed in chunks instead
p = B.build("m3_backbone_neck_int8", True, 500, method=CalibrationMethod.MinMax, tag="m3_pc_minmax", out_dir=work)
s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
r = evaluate(lambda x: s.run(None, {"images": x})[0][0], "calibmethod_minmax", None, out_dir=DAY2 / "results" / "dets_calibmethod", verbose=False)
res = {"minmax": {k: r[k] for k in ("mAP50-95", "mAP50")}}
(DAY2 / "results" / "calib_method.json").write_text(json.dumps(res, indent=2))
print("M3 per-channel ONNX, min-max calibration: mAP50-95", round(r["mAP50-95"], 4))
p.unlink()
