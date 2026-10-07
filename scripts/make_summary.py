"""Consolidate every Day 2 result into results/summary.json + summary.csv (one row per model)."""
import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
load = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else {}
acc, acc_t = load("accuracy.json"), load("accuracy_tflite.json")
lat, mem, prf = load("latency.json"), load("memory.json"), load("prf.json")
folds = load("folds_summary.json")

DESIGN = {"m0": "M0 FP32 baseline", "m1": "M1 all INT8 (incl. head)", "m2": "M2 backbone INT8 | neck+head FP32",
          "m3": "M3 backbone+neck INT8 | head FP32"}
rows = []


def add(fmt, key, accrec, design, gran, boxes):
    a, L, M, P = accrec, lat.get(key), mem.get(key), prf[key]
    f = folds.get(key, {}).get("mAP50-95", {})
    rows.append({
        "format": fmt, "model": key, "design": DESIGN[design], "granularity": gran, "box_output": boxes,
        "mAP50-95": a["mAP50-95"], "mAP50": a["mAP50"], "AP_small": a["AP_small"], "AP_medium": a["AP_medium"], "AP_large": a["AP_large"],
        "precision@0.25": P["P@0.25"], "recall@0.25": P["R@0.25"], "F1@0.25": P["F1@0.25"],
        "best_F1": P["best_F1"], "best_F1_conf": P["best_F1_conf"],
        "size_mb": a["size_mb"],
        "latency_ms_median": L["latency_ms_median"] if L else None, "latency_ms_p95": L["latency_ms_p95"] if L else None,
        "latency_round_spread_pct": L["round_spread_pct"] if L else None,
        "mem_load_mb": M["load_mb"] if M else None, "mem_peak_added_mb": M["peak_mb"] if M else None,
        "mem_baseline_mb": M["baseline_mb"] if M else None,
        "diff_vs_fp32_ci95": f.get("diff_ci95"), "diff_significant": f.get("significant"),
    })


# ONNX
for key in ["m0_fp32", "m1_all_int8_pc", "m1_all_int8_pt", "m1_all_int8_pc_norm", "m1_all_int8_pt_norm",
            "m2_backbone_int8_pc", "m2_backbone_int8_pt", "m3_backbone_neck_int8_pc", "m3_backbone_neck_int8_pt"]:
    if key not in acc:
        continue
    d = key[:2]
    gran = "-" if d == "m0" else ("per-channel" if "_pc" in key else "per-tensor")
    add("ONNX", key, acc[key], d, gran, "0-1" if key.endswith("_norm") else "pixels")
# TFLite
for key in acc_t:
    d = key.split("_")[1]
    gran = "-" if d == "m0" else ("per-channel" if key.endswith("_pc") else "per-tensor")
    add("TFLite", key, acc_t[key], d, gran, "0-1")

fp32 = {r["format"]: r for r in rows if r["design"].startswith("M0")}
for r in rows:
    b = fp32.get(r["format"])
    if b:
        r["compression_ratio"] = b["size_mb"] / r["size_mb"]
        r["accuracy_retained_pct"] = r["mAP50-95"] / b["mAP50-95"] * 100
        if r["latency_ms_median"] and b["latency_ms_median"]:
            r["speedup_vs_fp32"] = b["latency_ms_median"] / r["latency_ms_median"]
(R / "summary.json").write_text(json.dumps(rows, indent=2))
cols = list(rows[0].keys()) + ["compression_ratio", "accuracy_retained_pct", "speedup_vs_fp32"]
cols = list(dict.fromkeys(cols))
with open(R / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> summary.json / summary.csv")
for r in rows:
    print(f"{r['format']:7}{r['model']:32}{r['mAP50-95']:.4f} ret {r.get('accuracy_retained_pct',0):5.1f}%  {r['size_mb']:5.2f}MB x{r.get('compression_ratio',0):.2f}  {r['latency_ms_median'] or 0:5.1f}ms  mem +{r['mem_peak_added_mb'] or 0:.0f}MB")
