"""Build INT8 variants of YOLOv5nu, calibrated on COCO val images (disjoint from the eval set)."""
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from quantize_yolov5 import ImageCalibReader

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "original" / "yolov5nu.onnx"
CALIB = ROOT / "data" / "coco" / "calib"
OUT = ROOT / "models" / "int8"
OUT.mkdir(parents=True, exist_ok=True)

HEAD = ["/model.24/"]
NECK_OUT = ["/model.17/", "/model.20/", "/model.23/"]   # C3 blocks feeding the three Detect branches
VARIANTS = {
    "v1_percentile_head": (CalibrationMethod.Percentile, HEAD),
    "v2_minmax_head": (CalibrationMethod.MinMax, HEAD),
    "v3_percentile9999_head": (CalibrationMethod.Percentile, HEAD),
    "v4_percentile_head_neck": (CalibrationMethod.Percentile, HEAD + NECK_OUT),
    "ctrl_percentile_all": (CalibrationMethod.Percentile, []),
}

pre = OUT / "_prep.onnx"
quant_pre_process(str(SRC), str(pre))
model = onnx.load(str(pre))
inp = model.graph.input[0].name

for name, (method, prefixes) in VARIANTS.items():
    out = OUT / f"{name}.onnx"
    if out.exists():
        continue
    exclude = [n.name for n in model.graph.node if any(n.name.startswith(p) for p in prefixes)]
    print(f"== {name}: {method.name}, {len(exclude)} nodes FP32", flush=True)
    reader = ImageCalibReader(CALIB, inp, 640, 100)
    quantize_static(str(pre), str(out), reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=method, nodes_to_exclude=exclude,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                                   **({"CalibPercentile": 99.99} if "9999" in name else {})})
    print(f"   saved {out.name} {out.stat().st_size/1e6:.1f} MB", flush=True)
