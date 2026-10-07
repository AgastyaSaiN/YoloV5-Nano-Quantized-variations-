"""Latency benchmark with drift control (replaces the single-pass benchmark_runtime*.py).

A single pass over the models is unreliable: laptop CPU frequency and temperature drift between runs (the same FP32
model measured 27 ms and 37 ms in two sessions). So the models are INTERLEAVED over several rounds in shuffled order,
each model in a fresh process per round (20 warm-up + 100 timed runs, batch 1, 4 threads). A model's figure is the
median of its per-round medians; the spread across rounds is reported so the reader can see the measurement noise.

    python scripts/benchmark_latency.py [--rounds 5]   -> results/latency.json   (run on an otherwise idle machine)
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DAY2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
THREADS, WARMUP, RUNS = 4, 20, 100


def child(kind, path):
    from common import ValImages, postprocess, preprocess
    data = ValImages()
    items = []
    for i in range(10):
        img = data.read(i)[1]
        x, r, pad = preprocess(img)
        items.append((x, r, pad, img.shape[:2]))
    if kind == "onnx":
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        run, scale = (lambda x: sess.run(None, {name: x})[0][0]), 1.0
    else:
        from tflite_runner import Pipeline
        pipe = Pipeline(path, THREADS)
        run, scale = pipe, 640.0
    for k in range(WARMUP):
        run(items[k % 10][0])
    lat, post = [], []
    for k in range(RUNS):
        x, r, pad, shp = items[k % 10]
        t = time.perf_counter(); out = run(x); lat.append(time.perf_counter() - t)
        if k < 30:
            t = time.perf_counter(); postprocess(np.asarray(out).reshape(-1, 85), r, pad, shp, scale); post.append(time.perf_counter() - t)
    ms = lambda v: float(np.median(v) * 1e3)
    print(json.dumps({"median": ms(lat), "mean": float(np.mean(lat) * 1e3), "std": float(np.std(lat) * 1e3),
                      "p95": float(np.percentile(lat, 95) * 1e3), "post_ms": ms(post)}))


def jobs():
    j = [("onnx", DAY2 / "models/fp32/yolov5n.onnx", "m0_fp32")]
    for n in ["m1_all_int8_pc_norm", "m1_all_int8_pt_norm", "m2_backbone_int8_pc", "m2_backbone_int8_pt", "m3_backbone_neck_int8_pc", "m3_backbone_neck_int8_pt"]:
        j.append(("onnx", DAY2 / f"models/onnx_int8/{n}.onnx", n))
    for d in ["m0_fp32", "m1_all_int8_pc", "m1_all_int8_pt", "m2_backbone_int8_pc", "m2_backbone_int8_pt", "m3_backbone_neck_int8_pc", "m3_backbone_neck_int8_pt"]:
        j.append(("tflite", DAY2 / f"models/tflite/{d}", "tflite_" + d))
    return j


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("KIND", "PATH"))
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        sys.exit()
    venv_py = str(DAY2 / ".venv" / "Scripts" / "python.exe")
    allj = jobs()
    rounds = {key: [] for _, _, key in allj}
    rng = random.Random(0)
    for rd in range(a.rounds):
        order = allj[:]
        rng.shuffle(order)
        for kind, path, key in order:
            r = subprocess.run([sys.executable if kind == "onnx" else venv_py, __file__, "--child", kind, str(path)], capture_output=True, text=True)
            if r.returncode:
                print("FAILED", key, r.stderr[-300:]); continue
            rounds[key].append(json.loads(r.stdout.strip().splitlines()[-1]))
        print(f"round {rd + 1}/{a.rounds} done", flush=True)
    out = {}
    for _, path, key in allj:
        rs = rounds[key]
        med = [r["median"] for r in rs]
        out[key] = {"latency_ms_median": float(np.median(med)), "latency_ms_p95": float(np.median([r["p95"] for r in rs])),
                    "latency_ms_std": float(np.median([r["std"] for r in rs])), "post_ms_median": float(np.median([r["post_ms"] for r in rs])),
                    "round_medians_ms": med, "round_spread_pct": float((max(med) - min(med)) / np.median(med) * 100), "rounds": len(rs),
                    "threads": THREADS, "runs_per_round": RUNS}
    (DAY2 / "results" / "latency.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"{k:34} {v['latency_ms_median']:6.1f} ms  (rounds: {min(v['round_medians_ms']):.1f}-{max(v['round_medians_ms']):.1f})")
