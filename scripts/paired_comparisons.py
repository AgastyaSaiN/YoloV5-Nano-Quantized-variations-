"""Paired comparisons between variants from per-fold mAP (results/folds.json) -> results/paired.json.
diff = A - B per fold; 95% t-interval over 10 folds. 'significant' = interval excludes 0."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
F = json.loads((R / "folds.json").read_text())
K, T95 = 10, 2.262


def cmp(a, b, label):
    d = np.array(F[a]["ap"]) - np.array(F[b]["ap"])
    se = d.std(ddof=1) / np.sqrt(K)
    lo, hi = d.mean() - T95 * se, d.mean() + T95 * se
    return {"comparison": label, "A": a, "B": b, "diff": float(d.mean()), "ci95": [float(lo), float(hi)], "significant": bool(lo > 0 or hi < 0)}


out = []
for fmt, pre in (("ONNX", ""), ("TFLite", "tflite_")):
    m1 = "m1_all_int8_{g}_norm" if fmt == "ONNX" else "m1_all_int8_{g}"
    for name, tmpl in (("M1", m1), ("M2", "m2_backbone_int8_{g}"), ("M3", "m3_backbone_neck_int8_{g}")):
        k = lambda g: (pre + tmpl.format(g=g))
        out.append(cmp(k("pc"), k("pt"), f"{fmt} {name}: per-channel minus per-tensor"))
    for g in ("pc", "pt"):
        out.append(cmp(pre + f"m2_backbone_int8_{g}", pre + f"m3_backbone_neck_int8_{g}", f"{fmt} {g}: M2 (neck FP32) minus M3 (neck INT8)"))
        out.append(cmp(pre + f"m3_backbone_neck_int8_{g}", pre + (f"m1_all_int8_{g}_norm" if fmt == "ONNX" else f"m1_all_int8_{g}"), f"{fmt} {g}: M3 (head FP32) minus M1 (head INT8)"))
for name in ("m1_all_int8", "m2_backbone_int8", "m3_backbone_neck_int8"):
    for g in ("pc", "pt"):
        suf = "_norm" if name == "m1_all_int8" else ""
        out.append(cmp(f"{name}_{g}{suf}", f"tflite_{name}_{g}", f"{name.split('_')[0].upper()} {g}: ONNX minus TFLite"))
(R / "paired.json").write_text(json.dumps(out, indent=2))
for o in out:
    print(f"{o['comparison']:55}{o['diff']:+.4f}  [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}]  {'SIGNIFICANT' if o['significant'] else 'not significant'}")
