"""Time single-file TFLite models with a given runtime (TensorFlow's interpreter or the newer LiteRT), same protocol as the benchmark (results go to results/tflite_runtime_probe.json):
4 threads pinned to logical CPUs 0,2,4,6, 20 warm-up + 100 timed calls, random 512x512 input.
    <python of the environment> scripts/probe_tflite_runtime.py tf|litert model1.tflite model2.tflite ...
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

p = psutil.Process(); p.cpu_affinity([0, 2, 4, 6]); p.nice(psutil.HIGH_PRIORITY_CLASS)
kind = sys.argv[1]
if kind == "tf":
    import tensorflow as tf
    Interp = tf.lite.Interpreter
else:
    from ai_edge_litert.interpreter import Interpreter as Interp
OUT = Path(__file__).resolve().parent.parent / "results" / "tflite_runtime_probe.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {}
x = np.random.RandomState(0).rand(1, 512, 512, 3).astype(np.float32)
for path in sys.argv[2:]:
    it = Interp(model_path=path, num_threads=4)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    xin = x
    if d["dtype"] != np.float32:                                    # integer-input model: quantize the picture once, outside the timed loop
        scale, zero = d["quantization"]
        xin = np.clip(np.round(x / scale) + zero, np.iinfo(d["dtype"]).min, np.iinfo(d["dtype"]).max).astype(d["dtype"])
    t = []
    for k in range(120):
        it.set_tensor(d["index"], xin)
        t0 = time.perf_counter(); it.invoke(); dt = time.perf_counter() - t0
        if k >= 20:
            t.append(dt)
    name = Path(path).parent.name
    res.setdefault(name, {})[kind] = {"median_ms": float(np.median(t) * 1e3), "p95_ms": float(np.percentile(t, 95) * 1e3)}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"{kind:7s} {path.split('/')[-2] if '/' in path else path:42s} median {np.median(t) * 1e3:6.1f} ms   p95 {np.percentile(t, 95) * 1e3:6.1f} ms", flush=True)
