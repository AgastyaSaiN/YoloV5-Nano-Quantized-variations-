"""Where does inference time go?  ONNX Runtime per-node profile, aggregated by network region and operation type.
For each node the MEDIAN duration over 40 inferences is taken, then summed. Run on an idle machine.
    python scripts/profile_ort.py   -> results/profile.json"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

DAY2 = Path(__file__).resolve().parent.parent
MODELS = {"FP32": DAY2 / "models/fp32/yolov5n.onnx",
          "M1 per-channel": DAY2 / "models/onnx_int8/m1_all_int8_pc_norm.onnx",
          "M2 per-channel": DAY2 / "models/onnx_int8/m2_backbone_int8_pc.onnx",
          "M3 per-channel": DAY2 / "models/onnx_int8/m3_backbone_neck_int8_pc.onnx"}
region = lambda n: ("other" if not (m := re.search(r"/model\.(\d+)/", n)) else "backbone" if int(m.group(1)) < 10 else "neck" if int(m.group(1)) < 24 else "head")
OPCLS = {"Conv": "conv", "QLinearConv": "conv", "ConvInteger": "conv", "FusedConv": "conv",
         "Sigmoid": "activation (sigmoid/mul)", "QLinearSigmoid": "activation (sigmoid/mul)", "Mul": "activation (sigmoid/mul)", "QLinearMul": "activation (sigmoid/mul)",
         "QuantizeLinear": "quantize/dequantize", "DequantizeLinear": "quantize/dequantize"}
x = np.random.RandomState(0).rand(1, 3, 640, 640).astype(np.float32)
out = {}
for label, path in MODELS.items():
    so = ort.SessionOptions(); so.enable_profiling = True; so.intra_op_num_threads = 4
    so.profile_file_prefix = str(DAY2 / "results" / "logs" / "ort_profile")
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    for _ in range(40):
        s.run(None, {"images": x})
    ev = json.loads(Path(s.end_profiling()).read_text())
    dur = defaultdict(list); meta = {}
    for e in ev:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            n = e["name"][:-12]; dur[n].append(e["dur"]); meta[n] = e["args"].get("op_name", "")
    by_region, by_op, total = defaultdict(float), defaultdict(float), 0.0
    for n, d in dur.items():
        t = float(np.median(d)) / 1000.0                      # ms per inference
        total += t; by_region[region(n)] += t; by_op[OPCLS.get(meta[n], "other ops")] += t
    out[label] = {"total_node_ms": total, "by_region_ms": dict(by_region), "by_op_ms": dict(by_op)}
    print(f"{label:16} sum of node times {total:5.1f} ms | " + ", ".join(f"{k} {v:4.1f}" for k, v in sorted(by_region.items())) + " | " + ", ".join(f"{k} {v:4.1f}" for k, v in sorted(by_op.items())), flush=True)
    Path(s.end_profiling() if False else "").exists()
(DAY2 / "results" / "profile.json").write_text(json.dumps(out, indent=2))
