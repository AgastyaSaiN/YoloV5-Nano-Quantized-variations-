"""One list of every benchmarked model, shared by the latency / memory / evaluation scripts.

key naming:  m0_fp32                       FP32 baseline (640 px), ONNX Runtime
             s512_<design>_<pc|pt>         INT8 at 512 px, ONNX Runtime (models/onnx_int8_512)
             tflite_<folder>               TFLite model folders (models/tflite): m0_fp32 (640 px FP32), s512_m0_fp32, t512_* (single-file INT8, _io8 = integer input/output)
             ov_m0_fp32, ov_f512_fp32      the FP32 ONNX files (640 px, 512 px) run by OpenVINO; f512_fp32 is the same 512 px FP32 model in ONNX Runtime
             ovq_f512_m<1-3>_mixed         INT8 at 512 px quantized with OpenVINO NNCF (models/ov_int8)
box_scale: the network input size for models whose box output is normalised to 0-1, 1 for pixel-scale boxes.
"""
from pathlib import Path

DAY2 = Path(__file__).resolve().parent.parent


def jobs(kinds=("onnx", "tflite", "ov")):
    j = [dict(kind="onnx", path=DAY2 / "models/fp32/yolov5n.onnx", key="m0_fp32", box_scale=1.0)]
    j.append(dict(kind="onnx", path=DAY2 / "models/fp32_512/f512_norm.onnx", key="f512_fp32", box_scale=512.0))         # FP32 at the same 512 px input as the INT8 models
    for p in sorted((DAY2 / "models/onnx_int8_512").glob("*.onnx")):
        j.append(dict(kind="onnx", path=p, key="s512_" + p.stem, box_scale=512.0))
    for d in sorted((DAY2 / "models/tflite").glob("*")):
        if (d / "manifest.json").exists():
            j.append(dict(kind="tflite", path=d, key="tflite_" + d.name, box_scale=float(next((s for s in ("512",) if s in d.name), 640))))
    for p in sorted((DAY2 / "models/ov_int8").glob("*.xml")):
        j.append(dict(kind="ov", path=p, key="ovq_" + p.stem, box_scale=512.0))
    j.append(dict(kind="ov", path=DAY2 / "models/fp32/yolov5n.onnx", key="ov_m0_fp32", box_scale=1.0))
    j.append(dict(kind="ov", path=DAY2 / "models/fp32_512/f512_norm.onnx", key="ov_f512_fp32", box_scale=512.0))
    return [x for x in j if x["kind"] in kinds]
