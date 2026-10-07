"""Paired confidence intervals: is model X really different from the FP32 baseline?

The 5000 val images are split into 10 disjoint folds (image k -> fold k % 10). Each model's saved detections are
scored per fold with pycocotools; the per-fold DIFFERENCE vs the baseline is averaged and given a 95% t-interval
(9 degrees of freedom). If the interval contains 0, the difference is not distinguishable from noise.

    python scripts/fold_stats.py                # every results/dets/*.json
    python scripts/fold_stats.py name1 name2
Output: results/folds.json
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANN, DAY2  # noqa: E402

BASE, K, T95 = "m0_fp32", 10, 2.262
dets_dir = DAY2 / "results" / "dets"
names = sys.argv[1:] or sorted(p.stem for p in dets_dir.glob("*.json"))
if BASE not in names:
    names = [BASE] + names

with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO(str(ANN))
ids = sorted(coco.getImgIds())
folds = [ids[i::K] for i in range(K)]
out_path = DAY2 / "results" / "folds.json"
res = json.loads(out_path.read_text()) if out_path.exists() else {}


def per_fold(name):
    if not json.loads((dets_dir / f"{name}.json").read_text()):      # collapsed model: no detections at all
        return [0.0] * K, [0.0] * K
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(str(dets_dir / f"{name}.json"))
    ap, ap50 = [], []
    for f in folds:
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(coco, dt, "bbox"); E.params.imgIds = f
            E.evaluate(); E.accumulate(); E.summarize()
        ap.append(float(E.stats[0])); ap50.append(float(E.stats[1]))
    return ap, ap50


for n in names:
    if n not in res:
        print("scoring folds:", n, flush=True)
        ap, ap50 = per_fold(n)
        res[n] = {"ap": ap, "ap50": ap50}
        out_path.write_text(json.dumps(res, indent=2))

base = res[BASE]
summary = {}
for n, r in res.items():
    s = {}
    for key, label in (("ap", "mAP50-95"), ("ap50", "mAP50")):
        v = np.array(r[key])
        d = v - np.array(base[key])
        se = d.std(ddof=1) / np.sqrt(K)
        s[label] = {"mean_over_folds": float(v.mean()), "fold_std": float(v.std(ddof=1)),
                    "diff_vs_fp32": float(d.mean()), "diff_ci95": [float(d.mean() - T95 * se), float(d.mean() + T95 * se)],
                    "significant": bool(n != BASE and (d.mean() - T95 * se > 0 or d.mean() + T95 * se < 0))}
    summary[n] = s
(DAY2 / "results" / "folds_summary.json").write_text(json.dumps(summary, indent=2))
print(f"{'model':32}{'mAP50-95':>10}{'diff vs FP32':>14}{'95% CI':>22}  significant")
for n, s in summary.items():
    m = s["mAP50-95"]
    print(f"{n:32}{m['mean_over_folds']:10.4f}{m['diff_vs_fp32']:+14.4f}   [{m['diff_ci95'][0]:+.4f}, {m['diff_ci95'][1]:+.4f}]  {m['significant']}")
