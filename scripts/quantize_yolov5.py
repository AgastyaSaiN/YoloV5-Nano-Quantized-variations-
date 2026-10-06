"""
Static INT8 quantization of YOLOv5 (ONNX) that keeps the Detect head in FP32.

Setup:  pip install onnx onnxruntime onnxslim opencv-python
Usage:  python quantize_yolov5.py --weights yolov5s.pt --calib-dir path/to/images
        (or --weights model.onnx to skip the export step)

How the head is kept in float:
  Every ONNX node name looks like "/model.24/m.0/Conv". "model.24" is the Detect
  module (the last one). All nodes under that prefix go into nodes_to_exclude, so
  ORT leaves them FP32 and inserts DequantizeLinear on the way in.
  The last backbone/neck conv before the head can be excluded too (--extra-exclude).
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process


def export_onnx(weights: Path, imgsz: int) -> Path:
    """Export .pt -> .onnx. Works for original yolov5 repo checkpoints and ultralytics yolov5*u."""
    out = weights.with_suffix(".onnx")
    try:  # ultralytics package
        from ultralytics import YOLO
        YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=13, simplify=True, dynamic=False)
    except Exception as e:  # original ultralytics/yolov5 checkpoint -> use its export.py
        raise SystemExit(f"Auto export failed ({e}).\nFrom the yolov5 repo run:\n"
                         f"  python export.py --weights {weights} --include onnx --opset 13 --imgsz {imgsz}")
    return out



def to_fp32(path: Path) -> Path:
    """If the model is FP16, convert to FP32 (initializers, Cast targets, tensor types, constants)."""
    from onnx import TensorProto as T, numpy_helper
    m = onnx.load(str(path))
    if m.graph.input[0].type.tensor_type.elem_type != T.FLOAT16:
        return path
    for init in m.graph.initializer:
        if init.data_type == T.FLOAT16:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name))
    for n in m.graph.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == T.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32)))
            if n.op_type == "Cast" and a.name == "to" and a.i == T.FLOAT16:
                a.i = T.FLOAT
    for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        if vi.type.tensor_type.elem_type == T.FLOAT16:
            vi.type.tensor_type.elem_type = T.FLOAT
    out = path.with_name(path.stem + "_fp32.onnx")
    onnx.save(m, str(out))
    print(f"Converted FP16 -> FP32: {out.name}")
    return out

def find_head_prefixes(model: onnx.ModelProto) -> list[str]:
    """Detect is the highest-numbered '/model.N/' block in the graph."""
    idx = {int(m.group(1)) for n in model.graph.node if (m := re.match(r"/model\.(\d+)/", n.name))}
    if not idx:
        raise SystemExit("Could not find '/model.N/' node names. Pass --head-prefix manually.")
    return [f"/model.{max(idx)}/"]


class ImageCalibReader(CalibrationDataReader):
    def __init__(self, folder: Path, input_name: str, imgsz: int, limit: int):
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:limit]
        if not files:
            raise SystemExit(f"No images found in {folder}")
        self.files, self.name, self.imgsz = files, input_name, imgsz
        self.it = iter(self.files)

    def _prep(self, path):
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)                     # letterbox like YOLOv5
        img = cv2.resize(img, (int(round(w * r)), int(round(h * r))))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        x = canvas[:, :, ::-1].transpose(2, 0, 1)      # BGR->RGB, HWC->CHW
        return (x.astype(np.float32) / 255.0)[None]

    def get_next(self):
        for p in self.it:
            x = self._prep(p)
            if x is not None:
                return {self.name: x}
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt or .onnx")
    ap.add_argument("--calib-dir", required=True, help="folder of ~100-500 representative images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--head-prefix", nargs="*", help="override auto-detected head node prefix, e.g. /model.24/")
    ap.add_argument("--extra-exclude", nargs="*", default=[], help="extra node-name prefixes to keep FP32, e.g. /model.23/")
    ap.add_argument("--per-channel", action="store_true", default=True)
    args = ap.parse_args()

    w = Path(args.weights)
    onnx_fp32 = w if w.suffix == ".onnx" else export_onnx(w, args.imgsz)

    onnx_fp32 = to_fp32(onnx_fp32)
    pre = onnx_fp32.with_name(onnx_fp32.stem + "_prep.onnx")
    quant_pre_process(str(onnx_fp32), str(pre))        # shape inference + fusion-friendly cleanup
    model = onnx.load(str(pre))

    prefixes = (args.head_prefix or find_head_prefixes(model)) + args.extra_exclude
    exclude = [n.name for n in model.graph.node if any(n.name.startswith(p) for p in prefixes)]
    print(f"Keeping {len(exclude)} nodes in FP32 (prefixes: {prefixes})")

    out = onnx_fp32.with_name(onnx_fp32.stem + "_int8.onnx")
    reader = ImageCalibReader(Path(args.calib_dir), model.graph.input[0].name, args.imgsz, args.n_calib)
    quantize_static(
        str(pre), str(out), reader,
        quant_format=QuantFormat.QDQ,                  # QDQ: best compat (TensorRT/OpenVINO/CPU)
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        calibrate_method=CalibrationMethod.Percentile,  # more robust than MinMax for YOLO's outliers
        nodes_to_exclude=exclude,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    print(f"Saved {out}  ({out.stat().st_size/1e6:.1f} MB vs {onnx_fp32.stat().st_size/1e6:.1f} MB FP32)")


if __name__ == "__main__":
    main()
