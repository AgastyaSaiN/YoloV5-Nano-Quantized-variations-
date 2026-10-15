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

    size = int(next((t for t in ("512", "448", "416") if t in path), 640))      # reduced-resolution models take a smaller square input
    x = np.random.RandomState(0).rand(1, 3, size, size).astype(np.float32)
    if kind == "onnx":
        import onnxruntime as ort
    elif kind == "ov":
        import openvino as ov
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
    elif kind == "ov":
        comp = ov.Core().compile_model(path, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": THREADS})
        run = lambda: comp(x)
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
    ap.add_argument("--only", default="", help="comma-separated model keys (default: all)")
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        sys.exit()
    import registry
    jobs = [(j["kind"], j["path"], j["key"]) for j in registry.jobs() if not a.only or j["key"] in a.only.split(",")]
    out = DAY2 / "results" / "memory.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    try:
        import ai_edge_litert  # noqa: F401
        venv_py = sys.executable
    except ImportError:
        venv_py = str(DAY2 / ".venv" / "Scripts" / "python.exe")
    for kind, path, key in jobs:
        if key in res:
            continue
        py = venv_py if kind == "tflite" else sys.executable
        r = subprocess.run([py, __file__, "--child", kind, str(path)], capture_output=True, text=True)
        if r.returncode:
            print("FAILED", key, r.stderr[-400:])
            continue
        res[key] = json.loads(r.stdout.strip().splitlines()[-1])
        out.write_text(json.dumps(res, indent=2))
        d = res[key]
        print(f"{key:34} load +{d['load_mb']:6.1f} MB   peak +{d['peak_mb']:6.1f} MB   (baseline {d['baseline_mb']:.0f} MB)", flush=True)
