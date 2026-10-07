"""Is the Detect head's weight data identical to the original in every FP32-head variant?
Checks the three head convolution weight tensors of each model against the original FP32 ONNX model, and counts
quantize/dequantize operations inside the head (ONNX). Run with the TensorFlow venv:
    .venv/Scripts/python.exe scripts/check_head_untouched.py   -> results/head_check.json"""
import glob
import json
from pathlib import Path

import numpy as np
import onnx
import tensorflow as tf
from onnx import numpy_helper as nh

DAY2 = Path(__file__).resolve().parent.parent
orig = onnx.load(str(DAY2 / "models/fp32/yolov5n.onnx"))
init = {i.name: nh.to_array(i) for i in orig.graph.initializer}
ref = [init[n.input[1]] for n in orig.graph.node if n.name.startswith("/model.24/") and n.op_type == "Conv"]   # OIHW
ref_tfl = [w.transpose(0, 2, 3, 1) for w in ref]                                                              # OHWI (TFLite layout)
res = {}

for p in [DAY2 / "models/fp32/yolov5n_norm.onnx"] + [Path(x) for x in sorted(glob.glob(str(DAY2 / "models/onnx_int8/m*.onnx")))]:
    m = onnx.load(str(p))
    ini = {i.name: nh.to_array(i) for i in m.graph.initializer if i.data_type == 1}
    heads = [ini.get(n.input[1]) for n in m.graph.node if n.name.startswith("/model.24/") and n.op_type == "Conv"]
    same = sum(1 for w, r in zip(heads, ref) if w is not None and w.shape == r.shape and np.array_equal(w, r))
    qdq = sum(n.op_type in ("QuantizeLinear", "DequantizeLinear") and ("/model.24/" in n.name + n.input[0] + n.output[0]) for n in m.graph.node)
    res[p.stem] = {"head_weights_identical_fp32": same, "of": len(ref), "qdq_nodes_in_head": qdq}


HEAD_SHAPES = {tuple(w.shape) for w in ref_tfl}


def consts(folder, stage):
    # no default delegates: reading stored weights after the XNNPACK delegate has taken tensors over can crash
    it = tf.lite.Interpreter(str(DAY2 / f"models/tflite/{folder}/stage{stage}.tflite"),
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors(); out = []
    for t in it.get_tensor_details():
        if t["dtype"] in (np.float32, np.int8) and tuple(t["shape"]) in HEAD_SHAPES:     # only head-shaped 4-D weight tensors
            try:
                out.append(it.get_tensor(t["index"]).copy())
            except ValueError:
                pass
    return out


for f, st in [("m0_fp32", 1), ("m2_backbone_int8_pc", 2), ("m2_backbone_int8_pt", 2), ("m3_backbone_neck_int8_pc", 2),
              ("m3_backbone_neck_int8_pt", 2), ("m1_all_int8_pc", 1), ("m1_all_int8_pt", 1)]:
    cs = consts(f, st)
    res["tflite_" + f] = {"head_weights_identical_fp32": sum(any(a.dtype == np.float32 and a.shape == w.shape and np.array_equal(a, w) for a in cs) for w in ref_tfl),
                          "of": len(ref_tfl),
                          "head_shaped_int8_weights": sum(any(a.dtype == np.int8 and a.shape == w.shape for a in cs) for w in ref_tfl)}
(DAY2 / "results" / "head_check.json").write_text(json.dumps(res, indent=2))
for k, v in res.items():
    print("RESULT", f"{k:34}", v)
