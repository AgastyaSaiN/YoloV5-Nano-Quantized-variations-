"""How OpenVINO INT8 speed scales with cores and with throughput mode. Fresh process per setting, cores pinned to distinct physical cores
(logical CPUs 0,2,4,...), 20 warm-up + 100 timed calls, batch 1. Throughput mode runs several images in flight at once and reports ms per image.
  python scripts/probe_scaling.py -> results/scaling.json"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DAY2 = Path(__file__).resolve().parent.parent
MODELS = ["ovq_m3_mixed", "ovq_hybrid_s80_m3_mixed", "ovq_hybrid_s80_m1_mixed"]


def child(model, threads, mode):
    import openvino as ov
    import psutil
    psutil.Process().cpu_affinity(list(range(0, 2 * threads, 2)))
    psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
    x = np.random.RandomState(0).rand(1, 3, 640, 640).astype(np.float32)
    cfg = {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY" if mode == "latency" else "THROUGHPUT"}
    comp = ov.Core().compile_model(str(DAY2 / "models/ov_int8" / f"{model[4:]}.xml"), "CPU", cfg)
    if mode == "latency":
        for _ in range(20):
            comp(x)
        t = []
        for _ in range(100):
            t0 = time.perf_counter(); comp(x); t.append(time.perf_counter() - t0)
        print(json.dumps({"ms_per_image": float(np.median(t) * 1e3)}))
    else:
        q = ov.AsyncInferQueue(comp, comp.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
        for _ in range(30):
            q.start_async({0: x})
        q.wait_all()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            q.start_async({0: x})
        q.wait_all()
        print(json.dumps({"ms_per_image": (time.perf_counter() - t0) / n * 1e3, "streams": comp.get_property("NUM_STREAMS")}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4]); sys.exit()
    res = {}
    for m in MODELS:
        for mode, ths in (("latency", (2, 4, 6)), ("throughput", (4, 6))):
            for th in ths:
                vals = []
                for _ in range(3):
                    r = subprocess.run([sys.executable, __file__, "--child", m, str(th), mode], capture_output=True, text=True)
                    vals.append(json.loads(r.stdout.strip().splitlines()[-1]))
                res[f"{m}|{mode}|{th}"] = {"ms_per_image": float(np.median([v["ms_per_image"] for v in vals])), **{k: v for k, v in vals[0].items() if k != "ms_per_image"}}
                print(f"{m:28s} {mode:10s} {th} cores: {res[f'{m}|{mode}|{th}']['ms_per_image']:.2f} ms/image", flush=True)
    (DAY2 / "results" / "scaling.json").write_text(json.dumps(res, indent=2))
