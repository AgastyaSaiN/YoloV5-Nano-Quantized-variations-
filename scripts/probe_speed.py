"""Speed probe: how fast can each activation function go on each runtime? (accuracy-agnostic; models may be untrained)
Interleaved rounds, 4 threads pinned to distinct physical cores, batch 1 - same protocol as benchmark_latency.py.
    python scripts/probe_speed.py [--rounds 3]  -> results/speed_probe.json
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

DAY2 = Path(__file__).resolve().parent.parent
R = DAY2 / "models" / "relu"
S = R / "_speedtest"
MODELS = [  # label, runtime, onnx path
    ("FP32 SiLU (baseline)", "models/fp32/yolov5n.onnx"), ("FP32 Hardswish", "models/relu/hardswish_untrained_norm.onnx"), ("FP32 ReLU", "models/relu/relu_untrained_norm.onnx"),
    ("INT8 SiLU M2", "models/onnx_int8/m2_backbone_int8_pc.onnx"), ("INT8 SiLU M3", "models/onnx_int8/m3_backbone_neck_int8_pc.onnx"), ("INT8 SiLU M1", "models/onnx_int8/m1_all_int8_pc_norm.onnx"),
    ("INT8 Hardswish M2", "models/relu/_speedtest/hswish_m2.onnx"), ("INT8 Hardswish M3", "models/relu/_speedtest/hswish_m3.onnx"), ("INT8 Hardswish M1", "models/relu/_speedtest/hswish_m1.onnx"),
    ("INT8 ReLU M2", "models/relu/_speedtest/m2_backbone_int8_pc_speedtest.onnx"), ("INT8 ReLU M3", "models/relu/_speedtest/m3_backbone_neck_int8_pc_speedtest.onnx"), ("INT8 ReLU M1", "models/relu/_speedtest/m1_all_int8_pc_speedtest.onnx"),
]
ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()
jobs = [(l, rt, p) for l, p in MODELS if (DAY2 / p).exists() for rt in ("onnx", "ov")]
res = {(l, rt): [] for l, rt, p in jobs}
rng = random.Random(0)
for rd in range(a.rounds):
    order = jobs[:]; rng.shuffle(order)
    for l, rt, p in order:
        r = subprocess.run([sys.executable, str(DAY2 / "scripts/benchmark_latency.py"), "--child", rt, str(DAY2 / p), "640"], capture_output=True, text=True)
        if r.returncode == 0:
            res[(l, rt)].append(json.loads(r.stdout.strip().splitlines()[-1])["median"])
        else:
            print("FAILED", l, rt, r.stderr[-200:])
base = np.median(res[("FP32 SiLU (baseline)", "onnx")])
print(f"{'model':24}{'ONNX Runtime':>14}{'OpenVINO':>12}   (median ms; speed-up vs FP32 SiLU on ONNX Runtime in brackets)")
out = {}
for l, p in MODELS:
    row = []
    for rt in ("onnx", "ov"):
        v = res.get((l, rt))
        row.append(f"{np.median(v):6.1f} ({base / np.median(v):.2f}x)" if v else "      -      ")
        if v: out[f"{l}|{rt}"] = {"median_ms": float(np.median(v)), "rounds": v}
    print(f"{l:24}{row[0]:>14}{row[1]:>18}")
(DAY2 / "results" / "speed_probe.json").write_text(json.dumps(out, indent=2))
