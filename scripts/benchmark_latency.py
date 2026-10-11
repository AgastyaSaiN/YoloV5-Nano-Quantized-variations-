"""Latency benchmark (ONNX Runtime, TFLite, OpenVINO) with CPU pinning and drift control.

Protocol (also written to results/environment.json):
  * batch size 1; input 1x3x640x640 float32; pre- and post-processing are timed separately and NOT in the headline figure
  * 4 threads, the process pinned to logical CPUs 0,2,4,6 = four distinct physical cores of the 6-core i5-11400H
    (SMT siblings idle), process priority high. Unpinned threads were found to land on hyper-threaded siblings and
    inflated the FP32 baseline by ~50%.
  * per measurement: fresh process, 20 warm-up calls (first-call overhead is gone after 1-2 calls; see
    results/speed_diagnostics.json), then 100 timed calls
  * rounds: every model is measured once per round, models interleaved in shuffled order, 9 rounds, so CPU frequency /
    temperature drift (Windows 'Balanced' power plan) hits every model alike
  * headline figure = median of the per-round medians; the round-to-round spread is reported

    python scripts/benchmark_latency.py [--rounds 9] [--only key1,key2] [--kinds onnx,tflite,ov]   -> results/latency.json
Run on an otherwise idle machine, ideally on AC power.
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
THREADS, CPUS, WARMUP, RUNS = 4, [0, 2, 4, 6], 20, 100


def child(kind, path, box_scale):
    import psutil
    p = psutil.Process()
    p.cpu_affinity(CPUS)
    p.nice(psutil.HIGH_PRIORITY_CLASS)
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
        run = lambda x: sess.run(None, {name: x})[0][0]
    elif kind == "ov":
        import openvino as ov
        comp = ov.Core().compile_model(path, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": THREADS})
        run = lambda x: next(iter(comp(x).values()))[0]
    else:
        from tflite_runner import Pipeline
        pipe = Pipeline(path, THREADS)
        run = pipe
    for k in range(WARMUP):
        run(items[k % 10][0])
    lat, post = [], []
    for k in range(RUNS):
        x, r, pad, shp = items[k % 10]
        t = time.perf_counter(); out = run(x); lat.append(time.perf_counter() - t)
        if k < 30:
            t = time.perf_counter(); postprocess(np.asarray(out).reshape(-1, 85), r, pad, shp, float(box_scale)); post.append(time.perf_counter() - t)
    print(json.dumps({"median": float(np.median(lat) * 1e3), "mean": float(np.mean(lat) * 1e3), "std": float(np.std(lat) * 1e3),
                      "p95": float(np.percentile(lat, 95) * 1e3), "post_ms": float(np.median(post) * 1e3)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=3, metavar=("KIND", "PATH", "BOX_SCALE"))
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--only", default="")
    ap.add_argument("--kinds", default="onnx,tflite,ov")
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        sys.exit()
    from registry import jobs
    allj = [j for j in jobs(tuple(a.kinds.split(","))) if not a.only or j["key"] in a.only.split(",")]
    venv_py = str(DAY2 / ".venv" / "Scripts" / "python.exe")
    rounds = {j["key"]: [] for j in allj}
    rng = random.Random(0)
    for rd in range(a.rounds):
        order = allj[:]
        rng.shuffle(order)
        for j in order:
            r = subprocess.run([venv_py if j["kind"] == "tflite" else sys.executable, __file__, "--child", j["kind"], str(j["path"]), str(j["box_scale"])], capture_output=True, text=True)
            if r.returncode:
                print("FAILED", j["key"], r.stderr[-300:]); continue
            rounds[j["key"]].append(json.loads(r.stdout.strip().splitlines()[-1]))
        print(f"round {rd + 1}/{a.rounds} done", flush=True)
    out = {}
    for j in allj:
        rs = rounds[j["key"]]
        if not rs:
            continue
        med = [r["median"] for r in rs]
        out[j["key"]] = {"kind": j["kind"], "latency_ms_median": float(np.median(med)), "latency_ms_p95": float(np.median([r["p95"] for r in rs])),
                         "latency_ms_std": float(np.median([r["std"] for r in rs])), "post_ms_median": float(np.median([r["post_ms"] for r in rs])),
                         "round_medians_ms": med, "round_spread_pct": float((max(med) - min(med)) / np.median(med) * 100), "rounds": len(rs),
                         "threads": THREADS, "cpus": CPUS, "runs_per_round": RUNS, "warmup_per_round": WARMUP}
    path = DAY2 / "results" / "latency.json"
    prev = json.loads(path.read_text()) if path.exists() and a.only else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=2))
    for k, v in out.items():
        print(f"{k:38} {v['latency_ms_median']:6.1f} ms  (rounds {min(v['round_medians_ms']):.1f}-{max(v['round_medians_ms']):.1f}, spread {v['round_spread_pct']:.0f}%)")
