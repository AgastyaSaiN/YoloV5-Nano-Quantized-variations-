"""Build the TFLite M1 / M2 / M3 designs as ONE .tflite file each, the standard way (TensorFlow Lite post-training full-integer quantization
with a representative dataset; per-channel weights by default, per-tensor on request; layers that must stay float are excluded with the
quantization debugger's `denylisted_nodes`, which keeps them in the same file instead of splitting the network into several files).

Run with the TensorFlow venv:
    .venv\\Scripts\\python.exe scripts/build_tflite_single.py --size 512 [m1_all_int8_pc m3_backbone_neck_int8_pt ...]
Needs  models/tflite/_work/<size>/saved/{full,backbone,backbone_neck}  (made by split_onnx.py + convert_parts_onnx2tf.sh).
Writes  models/tflite/t<size>_<design>/stage1.tflite + manifest.json   (a one-stage pipeline, readable by tflite_runner.py).

Which ops stay float is found from the network's structure, not from names (the converted graph loses the layer names):
  M2: everything that is not an ancestor of the three backbone outputs (layers 4, 6, 9)  -> neck + head float
  M3: everything that is not an ancestor of the three neck outputs (layers 17, 20, 23)   -> head float
The three boundary tensors are located in the float graph by comparing values with the separately converted backbone / backbone+neck models.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CALIB_DIR, DAY2, preprocess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*", help="e.g. m3_backbone_neck_int8_pc (default: all six)")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--n-calib", type=int, default=500)
ap.add_argument("--int-io", action="store_true", help="integer input and output (int8) as NPU deployments usually require; only for the all-INT8 design M1")
a = ap.parse_args()
SIZE = a.size
WORK = DAY2 / "models" / "tflite" / "_work" / str(SIZE)
SAVED = WORK / "saved"
OUT = DAY2 / "models" / "tflite"
DESIGNS = ["m1_all_int8", "m2_backbone_int8", "m3_backbone_neck_int8"]


def rep_data(n):
    files = sorted(CALIB_DIR.glob("*.jpg"))[:n]

    def gen():
        for f in files:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield [preprocess(img, SIZE)[0].transpose(0, 2, 3, 1).astype(np.float32)]      # NHWC
    return gen


def float_model():
    p = WORK / "full_float_analysis.tflite"
    if not p.exists():
        p.write_bytes(tf.lite.TFLiteConverter.from_saved_model(str(SAVED / "full")).convert())
    return p.read_bytes()


def run_float(model_bytes, x_nhwc, preserve=False):
    it = tf.lite.Interpreter(model_content=model_bytes, experimental_preserve_all_tensors=preserve)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x_nhwc)
    it.invoke()
    return it


def region_names():
    """{design: [tensor names of the ops that must stay float]} found from the graph structure."""
    cache = WORK / "float_regions.json"
    if cache.exists():
        return json.loads(cache.read_text())
    img = cv2.imdecode(np.fromfile(str(sorted(CALIB_DIR.glob("*.jpg"))[0]), np.uint8), cv2.IMREAD_COLOR)
    x = preprocess(img, SIZE)[0].transpose(0, 2, 3, 1).astype(np.float32)
    full = run_float(float_model(), x, preserve=True)
    details = {d["index"]: d for d in full.get_tensor_details()}
    ops = full._get_ops_details()
    producer = {t: op["index"] for op in ops for t in op["outputs"]}
    by_index = {op["index"]: op for op in ops}

    def boundary(part):                                   # tensors of the full graph that equal the outputs of a separately converted part
        m = (SAVED / part / f"{part}_float32.tflite").read_bytes()
        sub = run_float(m, x)
        found = []
        for o in sub.get_output_details():
            v = sub.get_tensor(o["index"])
            cand = [(float(np.abs(full.get_tensor(i) - v).max()), i) for i, d in details.items() if tuple(d["shape"]) == v.shape and i in producer and d["dtype"] == np.float32]
            err, idx = min(cand)                       # the same layer output in both conversions; small float differences from different op fusion
            assert err < 1e-2, f"boundary tensor of {part} not found in the full graph (closest differs by {err:.3g})"
            found.append(idx)
        return found

    def ancestors(tensors):
        seen_ops, stack = set(), list(tensors)
        while stack:
            t = stack.pop()
            if t in producer and producer[t] not in seen_ops:
                seen_ops.add(producer[t])
                stack.extend(by_index[producer[t]]["inputs"])
        return seen_ops

    out = {}
    for design, part in (("m2_backbone_int8", "backbone"), ("m3_backbone_neck_int8", "backbone_neck")):
        keep_int8 = ancestors(boundary(part))
        float_ops = [op for op in ops if op["index"] not in keep_int8]
        out[design] = sorted({details[t]["name"] for op in float_ops for t in op["outputs"]})
        print(f"{design}: {len(keep_int8)} ops quantized, {len(float_ops)} ops kept float", flush=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def build(design, per_channel):
    deny = [] if design.startswith("m1") else region_names()[design]
    conv = tf.lite.TFLiteConverter.from_saved_model(str(SAVED / "full"))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data(a.n_calib)
    if not per_channel:
        conv._experimental_disable_per_channel = True                  # per-tensor ("layer-wise") weights
    if not deny:
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]      # strict: any non-INT8 op fails loudly
        if a.int_io:
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        return conv.convert()
    dbg = tf.lite.experimental.QuantizationDebugger(converter=conv, debug_dataset=rep_data(1),
                                                    debug_options=tf.lite.experimental.QuantizationDebugOptions(denylisted_nodes=deny))
    return dbg.get_nondebug_quantized_model()


if __name__ == "__main__":
    todo = a.names or [f"{d}_{g}" for d in (DESIGNS[:1] if a.int_io else DESIGNS) for g in ("pc", "pt")]
    for name in todo:
        design, g = name.rsplit("_", 1)
        d = OUT / (f"t{SIZE}_{name}" if a.n_calib == 500 else f"t{SIZE}n{a.n_calib}_{name}") if not a.int_io else OUT / f"t{SIZE}_{name}_io8"        # other calibration-set sizes are for the study only
        if (d / "manifest.json").exists():
            print("skip (exists)", name)
            continue
        print("==", name, flush=True)
        blob = build(design, g == "pc")
        d.mkdir(parents=True, exist_ok=True)
        (d / "stage1.tflite").write_bytes(blob)
        size = len(blob) / 1e6
        (d / "manifest.json").write_text(json.dumps({"design": name, "stages": ["stage1.tflite"], "note": f"single-file {design}, {'per-channel' if g == 'pc' else 'per-tensor'}", "size_mb": size}, indent=2))
        print(f"   {name}: one file, {size:.2f} MB", flush=True)
