"""Speed diagnostics for the ONNX models (run on an otherwise idle machine; results -> results/speed_diagnostics.json).

  1. warm-up curve   : latency of the first 60 calls in a fresh process, to justify the warm-up count
  2. thread scaling  : median latency with 1, 2, 4, 6 threads (pinned to distinct physical cores) and 12 (all logical)
  3. op breakdown    : ONNX Runtime per-operation time (fused op names), top operations per model
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DAY2 = Path(__file__).resolve().parent.parent
MODELS = {"FP32": "models/fp32/yolov5n.onnx", "M1 per-channel": "models/onnx_int8/m1_all_int8_pc_norm.onnx",
          "M2 per-channel": "models/onnx_int8/m2_backbone_int8_pc.onnx", "M3 per-channel": "models/onnx_int8/m3_backbone_neck_int8_pc.onnx"}


def child(mode, path, threads):
    import onnxruntime as ort
    import psutil
    import time
    p = psutil.Process()
    if threads and threads < 12:
        p.cpu_affinity([2 * i for i in range(threads)])          # logical CPUs 0,2,4,.. = distinct physical cores
    x = np.random.RandomState(0).rand(1, 3, 640, 640).astype(np.float32)
    so = ort.SessionOptions(); so.intra_op_num_threads = threads or 4
    if mode == "ops":
        so.enable_profiling = True
    s = ort.InferenceSession(str(DAY2 / path), so, providers=["CPUExecutionProvider"])
    n = {"warmup": 60, "scale": 70, "ops": 40}[mode]
    lat = []
    for _ in range(n):
        t = time.perf_counter(); s.run(None, {"images": x}); lat.append((time.perf_counter() - t) * 1e3)
    if mode == "ops":
        ev = json.loads(Path(s.end_profiling()).read_text())
        d = defaultdict(list)
        for e in ev:
            if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
                d[(e["name"], e["args"].get("op_name", ""))].append(e["dur"] / 1000.0)
        by = defaultdict(float)
        for (name, op), v in d.items():
            by[op] += float(np.median(v))
        print(json.dumps({"by_op": dict(sorted(by.items(), key=lambda kv: -kv[1])[:10]), "sum": sum(by.values())}))
    elif mode == "warmup":
        print(json.dumps({"lat": lat}))
    else:
        print(json.dumps({"median": float(np.median(lat[10:])), "p95": float(np.percentile(lat[10:], 95))}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        sys.exit()
    run = lambda mode, path, th=4: json.loads(subprocess.run([sys.executable, __file__, "--child", mode, path, str(th)], capture_output=True, text=True).stdout.strip().splitlines()[-1])
    out = {"warmup": {}, "scaling": {}, "ops": {}}
    for k, p in MODELS.items():
        lat = np.array(run("warmup", p)["lat"])
        steady = float(np.median(lat[30:]))
        out["warmup"][k] = {"first_calls_ms": [round(float(v), 1) for v in lat[:10]], "steady_ms": steady,
                            "calls_until_within_10pct": int(next((i + 1 for i in range(len(lat)) if np.all(lat[i:i + 5] < steady * 1.10)), len(lat)))}
        print(f"warm-up {k:16} first calls {out['warmup'][k]['first_calls_ms'][:6]} steady {steady:.1f} ms; within 10% after call {out['warmup'][k]['calls_until_within_10pct']}", flush=True)
    for th in (1, 2, 4, 6, 12):
        out["scaling"][str(th)] = {k: run("scale", p, th) for k, p in MODELS.items()}
        print(f"threads {th:2d}: " + "  ".join(f"{k} {v['median']:.1f}" for k, v in out["scaling"][str(th)].items()), flush=True)
    for k, p in MODELS.items():
        out["ops"][k] = run("ops", p, 4)
        print(f"ops {k:16} total {out['ops'][k]['sum']:.1f} ms:", {a: round(b, 1) for a, b in list(out['ops'][k]['by_op'].items())[:6]}, flush=True)
    (DAY2 / "results" / "speed_diagnostics.json").write_text(json.dumps(out, indent=2))
