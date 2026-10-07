"""Build the ONNX INT8 variants (M1-M3) in per-channel (_pc) and per-tensor (_pt) form.

  M1: everything INT8 (backbone + neck + head)          -> true control
  M2: backbone INT8 | neck FP32 | head FP32
  M3: backbone INT8 | neck INT8 | head FP32

YOLOv5 v7.0 layers: backbone 0-9, neck (PANet) 10-23, head (Detect) 24. FP32 parts are excluded
from quantization by node-name prefix.

    python scripts/build_onnx_variants.py            # all six
    python scripts/build_onnx_variants.py m3_backbone_neck_int8_pc
"""
import argparse
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import HistogramCalibrater
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from common import CALIB_DIR, DAY2, preprocess  # noqa: E402

SRC = DAY2 / "models" / "fp32" / "yolov5n.onnx"          # pixel-scale box output (original export)
SRC_NORM = DAY2 / "models" / "fp32" / "yolov5n_norm.onnx"  # box output normalised to 0-1
OUT = DAY2 / "models" / "onnx_int8"
WORK = OUT / "_work"

BACKBONE = [f"/model.{i}/" for i in range(0, 10)]
NECK = [f"/model.{i}/" for i in range(10, 24)]
HEAD = ["/model.24/"]
DESIGNS = {                       # design name -> prefixes kept in FP32
    "m1_all_int8": [],
    "m2_backbone_int8": NECK + HEAD,
    "m3_backbone_neck_int8": HEAD,
}


class TrainCalibReader(CalibrationDataReader):
    """Letterboxed train2017 images, preprocessed exactly like the evaluation pipeline."""

    def __init__(self, input_name, n):
        files = sorted(CALIB_DIR.glob("*.jpg"))[:n]
        assert len(files) == n, f"only {len(files)} calibration images"
        self.name, self.it = input_name, iter(files)

    def get_next(self):
        for f in self.it:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return {self.name: preprocess(img)[0]}
        return None


def install_chunked_calibration(chunk=15, cls=HistogramCalibrater):
    """ORT keeps every image's outputs in RAM before merging (OOM at ~300 images in Day 1).
    Feed the histogram collector `chunk` images at a time instead; it merges histograms incrementally."""
    orig = cls.collect_data

    class Chunk:
        def __init__(self, reader, n):
            self.reader, self.n, self.exhausted = reader, n, False

        def get_next(self):
            if self.n == 0:
                return None
            self.n -= 1
            x = self.reader.get_next()
            if x is None:
                self.exhausted = True
            return x

    def chunked(self, reader):
        while True:
            c = Chunk(reader, chunk)
            try:
                orig(self, c)
            except ValueError:            # "No data is collected": previous chunk consumed the last image
                break
            if c.exhausted:
                break

    cls.collect_data = chunked


def install_calibration_cache(n_calib, method, src_stem="yolov5n"):
    """Calibration ranges do not depend on which layers are excluded or on per-channel vs per-tensor
    (ORT applies exclusions after calibrating), so compute them once per (n_images, method) and reuse."""
    import pickle
    WORK.mkdir(parents=True, exist_ok=True)
    cache = WORK / f"calib_ranges_{src_stem}_{n_calib}_{method.name}.pkl"
    orig_collect, orig_compute = HistogramCalibrater.collect_data, HistogramCalibrater.compute_data

    def collect(self, reader):
        if not cache.exists():
            orig_collect(self, reader)

    def compute(self):
        if cache.exists():
            return pickle.loads(cache.read_bytes())
        ranges = orig_compute(self)
        cache.write_bytes(pickle.dumps(ranges))
        return ranges

    HistogramCalibrater.collect_data, HistogramCalibrater.compute_data = collect, compute


def preprocessed_model(src=None):
    src = Path(src or SRC)
    WORK.mkdir(parents=True, exist_ok=True)
    pre = WORK / f"{src.stem}_prep.onnx"
    if not pre.exists():
        quant_pre_process(str(src), str(pre))
    return pre


def build(design, per_channel, n_calib=500, method=CalibrationMethod.Percentile, tag=None, exclude_prefixes=None, out_dir=None, src=None, extra_options=None):
    pre = preprocessed_model(src)
    model = onnx.load(str(pre))
    prefixes = DESIGNS[design] if exclude_prefixes is None else exclude_prefixes
    exclude = [n.name for n in model.graph.node if any(n.name.startswith(p) for p in prefixes)]
    name = tag or f"{design}_{'pc' if per_channel else 'pt'}"
    out = (out_dir or OUT) / f"{name}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"== {name}: {len(exclude)} nodes kept FP32, calib={n_calib} imgs, {'per-channel' if per_channel else 'per-tensor'}", flush=True)
    quantize_static(str(pre), str(out), TrainCalibReader(model.graph.input[0].name, n_calib),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=per_channel, calibrate_method=method, nodes_to_exclude=exclude,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True, **(extra_options or {})})
    print(f"   saved {out.name}  {out.stat().st_size / 1e6:.2f} MB", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="e.g. m3_backbone_neck_int8_pc (default: all six)")
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--norm", action="store_true", help="use the normalised-box model; output names get a _norm suffix")
    a = ap.parse_args()
    install_chunked_calibration()
    src = SRC_NORM if a.norm else SRC
    install_calibration_cache(a.n_calib, CalibrationMethod.Percentile, src.stem)
    todo = a.names or [f"{d}_{g}" for d in DESIGNS for g in ("pc", "pt")]
    for name in todo:
        design, g = name.rsplit("_", 1)
        tag = f"{name}_norm" if a.norm else name
        if (OUT / f"{tag}.onnx").exists():
            print("skip (exists)", tag)
            continue
        build(design, g == "pc", a.n_calib, tag=tag, src=src)
