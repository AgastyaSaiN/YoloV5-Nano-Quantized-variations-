"""Precision / recall / F1 from the saved detections, using the official COCO matching protocol (pycocotools):
greedy matching by score at IoU 0.5, crowd regions ignored, up to 300 detections per image, micro-averaged over all
classes and all 5000 images. A confidence sweep (0.05 to 0.95, step 0.05) gives P/R/F1 at 0.25 and the best F1.

    python scripts/compute_prf.py     -> results/prf.json
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

THRESH = np.round(np.arange(0.05, 0.96, 0.05), 2)
out = DAY2 / "results" / "prf.json"
res = json.loads(out.read_text()) if out.exists() else {}
with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO(str(ANN))
for f in sorted((DAY2 / "results" / "dets").glob("*.json")):
    name = f.stem
    if name in res:
        continue
    dets = json.loads(f.read_text())
    if not dets:                                    # collapsed model: nothing detected
        res[name] = {"P@0.25": 0.0, "R@0.25": 0.0, "F1@0.25": 0.0, "best_F1": 0.0, "best_F1_conf": 0.05, "curve": []}
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(coco, coco.loadRes(str(f)), "bbox")
        E.params.iouThrs = np.array([0.5]); E.params.maxDets = [1, 10, 300]
        E.params.areaRng = [[0, 1e10]]; E.params.areaRngLbl = ["all"]
        E.evaluate()
    sc, tp, ngt = [], [], 0
    for ev in E.evalImgs:
        if ev is None:
            continue
        ign = np.array(ev["dtIgnore"])[0].astype(bool)
        sc += list(np.array(ev["dtScores"])[~ign]); tp += list(np.array(ev["dtMatches"])[0][~ign] > 0)
        ngt += int((~np.array(ev["gtIgnore"]).astype(bool)).sum())
    sc, tp = np.array(sc), np.array(tp)
    curve = []
    for t in THRESH:
        s = sc >= t
        n = int(tp[s].sum()); p = n / max(int(s.sum()), 1); r = n / ngt
        curve.append({"conf": float(t), "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)})
    best = max(curve, key=lambda d: d["f1"]); at = next(d for d in curve if abs(d["conf"] - 0.25) < 1e-9)
    res[name] = {"P@0.25": at["precision"], "R@0.25": at["recall"], "F1@0.25": at["f1"],
                 "best_F1": best["f1"], "best_F1_conf": best["conf"], "curve": curve}
    out.write_text(json.dumps(res, indent=1))
    print(f"{name:34} P {at['precision']:.4f} R {at['recall']:.4f} F1 {at['f1']:.4f}   best F1 {best['f1']:.4f} @ {best['conf']}", flush=True)
out.write_text(json.dumps(res, indent=1))
