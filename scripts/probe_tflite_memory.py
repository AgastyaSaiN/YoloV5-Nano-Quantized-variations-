"""Where TFLite's memory goes: resident memory of a TFLite model after creating the interpreter, after allocating tensors, and after running,
with the runtime's default XNNPACK engine on and off (fresh process per measurement).  python scripts/probe_tflite_memory.py -> results/tflite_memory_probe.json"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

DAY2 = Path(__file__).resolve().parent.parent
MODELS = [("m0_fp32", 640), ("s512_m0_fp32", 512), ("t512_m1_all_int8_pc", 512), ("t512_m3_backbone_neck_int8_pc", 512)]


def child(path, size, mode):
    import gc

    import numpy as np
    import psutil
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    p = psutil.Process(); gc.collect()
    mb = lambda: p.memory_info().rss / 1e6
    base = mb(); peak = [base]; stop = threading.Event()

    def samp():
        while not stop.is_set():
            peak[0] = max(peak[0], mb()); time.sleep(0.002)
    threading.Thread(target=samp, daemon=True).start()
    kw = {"experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES} if mode == "no_xnnpack" else {}
    it = Interpreter(model_path=path, num_threads=4, **kw)
    it.allocate_tensors(); at_load = mb() - base
    d = it.get_input_details()[0]
    x = np.random.RandomState(0).rand(1, size, size, 3).astype(np.float32)
    for _ in range(31):
        it.set_tensor(d["index"], x); it.invoke()
    stop.set()
    print(json.dumps({"after_load_mb": at_load, "peak_mb": peak[0] - base}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4]); sys.exit()
    res = {}
    for name, size in MODELS:
        for mode in ("default", "no_xnnpack"):
            r = subprocess.run([sys.executable, __file__, "--child", str(DAY2 / "models/tflite" / name / "stage1.tflite"), str(size), mode], capture_output=True, text=True)
            res.setdefault(name, {})[mode] = json.loads(r.stdout.strip().splitlines()[-1])
            print(f"{name:32s} {mode:11s} load +{res[name][mode]['after_load_mb']:5.1f} MB  peak +{res[name][mode]['peak_mb']:5.1f} MB", flush=True)
    (DAY2 / "results" / "tflite_memory_probe.json").write_text(json.dumps(res, indent=2))
