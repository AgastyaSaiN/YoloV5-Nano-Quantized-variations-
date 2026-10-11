"""Quantize the SiLU YOLOv5n with OpenVINO's NNCF (post-training, no retraining), Detect head kept FP32.

Why: ONNX Runtime's QDQ quantizer puts quantize/dequantize steps between each convolution and its SiLU activation,
which blocks OpenVINO from fusing them. NNCF places the quantizers where OpenVINO's fused INT8 kernels expect them.

    python scripts/quantize_openvino.py [--preset mixed|performance] [--n 500]   -> models/ov_int8/<design>_<preset>.xml/.bin
Designs: m1 (all INT8), m2 (backbone INT8; neck + head FP32), m3 (backbone + neck INT8; head FP32).
"""
import argparse
import sys
from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CALIB_DIR, DAY2, preprocess  # noqa: E402

SRC = DAY2 / "models" / "fp32" / "yolov5n_norm.onnx"
OUT = DAY2 / "models" / "ov_int8"
IGNORE = {"m1": [], "m2": [r".*/model\.(1[0-9]|2[0-4])/.*"], "m3": [r".*/model\.24/.*"]}


def transform(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return preprocess(img)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mixed", choices=["mixed", "performance"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--designs", default="m1,m2,m3")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(CALIB_DIR.glob("*.jpg"))[:a.n]
    preset = {"mixed": nncf.QuantizationPreset.MIXED, "performance": nncf.QuantizationPreset.PERFORMANCE}[a.preset]
    for d in a.designs.split(","):
        model = ov.Core().read_model(a.src)
        scope = nncf.IgnoredScope(patterns=IGNORE[d]) if IGNORE[d] else None
        q = nncf.quantize(model, nncf.Dataset(files, transform), preset=preset, subset_size=len(files), ignored_scope=scope)
        path = out / f"{a.prefix}{d}_{a.preset}.xml"
        ov.save_model(q, str(path))
        print(f"saved {path.name}  ({path.with_suffix('.bin').stat().st_size / 1e6:.2f} MB weights)", flush=True)
