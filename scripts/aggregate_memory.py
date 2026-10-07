"""Combine the repeated memory runs (results/logs/memory_run{1,2,3}.json) into results/memory.json:
median of each figure, plus the min-max spread so the measurement noise is visible."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
runs = [json.loads(p.read_text()) for p in sorted((R / "logs").glob("memory_run*.json"))]
out = {}
for key in runs[0]:
    vals = [r[key] for r in runs if key in r]
    out[key] = {f: float(np.median([v[f] for v in vals])) for f in ("baseline_mb", "load_mb", "peak_mb", "steady_mb")}
    out[key]["peak_min_mb"] = float(min(v["peak_mb"] for v in vals))
    out[key]["peak_max_mb"] = float(max(v["peak_mb"] for v in vals))
    out[key]["runs"] = len(vals)
(R / "memory.json").write_text(json.dumps(out, indent=2))
for k, v in out.items():
    print(f"{k:34} peak +{v['peak_mb']:6.1f} MB  (runs {v['peak_min_mb']:.1f} to {v['peak_max_mb']:.1f})  load +{v['load_mb']:.1f}")
