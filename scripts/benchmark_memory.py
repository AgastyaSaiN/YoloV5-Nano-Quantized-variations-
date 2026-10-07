"""Memory benchmark: how much memory a model adds to a process, measured cleanly.

Each model runs in a FRESH process:
  1. libraries imported, one input tensor created, baseline resident memory (RSS) recorded
  2. model loaded                       -> "load" = RSS after load - baseline
  3. 5 warm-up + 30 inference runs, with a background sampler recording RSS every 2 ms
                                         -> "peak" = highest RSS - baseline  (model + activations + runtime buffers)
No other image buffers are kept, so the figures reflect the model rather than the test harness.

    python scripts/benchmark_memory.py          # all models -> results/memory.json
"""
import argparse
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

DAY2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
THREADS, WARMUP, RUNS = 4, 5, 30


def child(kind, path):
    import numpy as np
    import psutil

    x = np.random.RandomState(0).rand(1, 3, 640, 640).astype(np.float32)
    if kind == "onnx":
        import onnxruntime as ort
    else:
        from tflite_runner import Pipeline
    proc = psutil.Process()
    gc.collect()
    base = proc.memory_info().rss / 1e6
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss / 1e6)
            time.sleep(0.002)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    if kind == "onnx":
        so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        run = lambda: sess.run(None, {name: x})
    else:
        pipe = Pipeline(path, THREADS)
        run = lambda: pipe(x)
    loaded = proc.memory_info().rss / 1e6
    for _ in range(WARMUP + RUNS):
        run()
    end = proc.memory_info().rss / 1e6
    stop.set(); th.join()
    print(json.dumps({"baseline_mb": base, "load_mb": loaded - base, "peak_mb": max(peak) - base, "steady_mb": end - base}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("KIND", "PATH"))
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        sys.exit()
    jobs = [("onnx", DAY2 / "models/fp32/yolov5n.onnx", "m0_fp32")]
    for n in ["m1_all_int8_pc_norm", "m1_all_int8_pt_norm", "m2_backbone_int8_pc", "m2_backbone_int8_pt", "m3_backbone_neck_int8_pc", "m3_backbone_neck_int8_pt"]:
        jobs.append(("onnx", DAY2 / f"models/onnx_int8/{n}.onnx", n))
    for d in ["m0_fp32", "m1_all_int8_pc", "m1_all_int8_pt", "m2_backbone_int8_pc", "m2_backbone_int8_pt", "m3_backbone_neck_int8_pc", "m3_backbone_neck_int8_pt"]:
        jobs.append(("tflite", DAY2 / f"models/tflite/{d}", "tflite_" + d))
    out = DAY2 / "results" / "memory.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    venv_py = str(DAY2 / ".venv" / "Scripts" / "python.exe")
    for kind, path, key in jobs:
        if key in res:
            continue
        py = sys.executable if kind == "onnx" else venv_py
        r = subprocess.run([py, __file__, "--child", kind, str(path)], capture_output=True, text=True)
        if r.returncode:
            print("FAILED", key, r.stderr[-400:])
            continue
        res[key] = json.loads(r.stdout.strip().splitlines()[-1])
        out.write_text(json.dumps(res, indent=2))
        d = res[key]
        print(f"{key:34} load +{d['load_mb']:6.1f} MB   peak +{d['peak_mb']:6.1f} MB   (baseline {d['baseline_mb']:.0f} MB)", flush=True)
