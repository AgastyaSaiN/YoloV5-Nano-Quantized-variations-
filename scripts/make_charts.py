"""Day 2 charts (white background, report style, no titles baked in; captions live in report.html).
Reads results/summary.json, sensitivity.json, calib_size.json -> results/charts/*.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
OUT = R / "charts"
OUT.mkdir(exist_ok=True)
S = {(r["format"], r["model"]): r for r in json.loads((R / "summary.json").read_text())}
INK, GRID, GRAY = "#1a1a1a", "#e6e6e6", "#8a8f98"
C = {("ONNX", "pc"): "#1f4e9e", ("ONNX", "pt"): "#8fb3e8", ("TFLite", "pc"): "#1b7f5c", ("TFLite", "pt"): "#8fd4b8"}
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": "#bbbbbb", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})

# model keys per design (ONNX M1 uses the 0-1 box version; the pixel-scale one is shown separately)
D = [("M1  all INT8\n(head INT8 too)", "m1_all_int8", "_norm"), ("M2  backbone INT8\nneck + head FP32", "m2_backbone_int8", ""),
     ("M3  backbone + neck INT8\nhead FP32", "m3_backbone_neck_int8", "")]
key = lambda fmt, base, g, suf: (fmt, (f"{base}_{g}{suf}" if fmt == "ONNX" else f"tflite_{base}_{g}"))
base_map = S[("ONNX", "m0_fp32")]["mAP50-95"]


def style(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID)
    ax.set_axisbelow(True)


# 1 ---------------------------------------------------------------- accuracy by design
fig, ax = plt.subplots(figsize=(10, 5)); style(ax)
w = 0.2
for j, (fmt, g) in enumerate([("ONNX", "pc"), ("ONNX", "pt"), ("TFLite", "pc"), ("TFLite", "pt")]):
    vals = [S[key(fmt, b, g, s)]["mAP50-95"] for _, b, s in D]
    bars = ax.bar(np.arange(3) + (j - 1.5) * w, vals, w, color=C[(fmt, g)], label=f"{fmt}, {'per-channel' if g == 'pc' else 'per-tensor'}")
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + w / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8.5)
ax.axhline(base_map, color=GRAY, ls="--", lw=1.2)
ax.text(2.45, base_map - 0.0035, f"FP32 baseline {base_map:.3f}", color=GRAY, ha="right", fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels([d[0] for d in D]); ax.set_ylim(0.20, 0.315)
ax.set_ylabel("mAP50-95 (COCO val2017, 5000 images)"); ax.legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "1_accuracy.png", dpi=160); plt.close(fig)

# 2 ---------------------------------------------------------------- size vs accuracy
fig, ax = plt.subplots(figsize=(10, 5.4)); style(ax); style(ax, "x")
ax.scatter([S[("ONNX", "m0_fp32")]["size_mb"]], [base_map], s=110, color=GRAY, zorder=3)
ax.annotate("FP32", (S[("ONNX", "m0_fp32")]["size_mb"], base_map), textcoords="offset points", xytext=(-8, 8), ha="right")
for fmt in ("ONNX", "TFLite"):
    for lab, b, s in D:
        for g in ("pc", "pt"):
            r = S[key(fmt, b, g, s)]
            ax.scatter(r["size_mb"], r["mAP50-95"], s=85, color=C[(fmt, g)], edgecolor="white", zorder=3)
            ax.annotate(lab.split("  ")[0] + ("" if fmt == "ONNX" else "t"), (r["size_mb"], r["mAP50-95"]),
                        textcoords="offset points", xytext=(6, 4 if g == "pc" else -11), fontsize=8.5, color=C[(fmt, g)])
ax.set_xlabel("Model size (MB, all pipeline stages)"); ax.set_ylabel("mAP50-95")
ax.set_xlim(8.4, 1.6)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C[k], label=f"{k[0]} {'per-channel' if k[1] == 'pc' else 'per-tensor'}") for k in C],
          frameon=False, loc="lower left")
ax.text(0.99, 0.97, "up and to the right = smaller and more accurate\n(\"t\" suffix = TFLite)", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=GRAY)
fig.tight_layout(); fig.savefig(OUT / "2_size_vs_accuracy.png", dpi=160); plt.close(fig)

# 3 ---------------------------------------------------------------- latency + memory
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5)); style(a1); style(a2)
names, lat, p95, mem, cols = [], [], [], [], []
for fmt in ("ONNX", "TFLite"):
    base = S[(fmt, "m0_fp32" if fmt == "ONNX" else "tflite_m0_fp32")]
    items = [("FP32", base, GRAY)]
    for lab, b, s in D:
        for g in ("pc", "pt"):
            items.append((f"{lab.split('  ')[0]} {g}", S[key(fmt, b, g, s)], C[(fmt, g)]))
    for n, r, c in items:
        names.append(f"{n}\n{fmt}"); lat.append(r["latency_ms_median"]); p95.append(r["latency_ms_p95"])
        mem.append(r["mem_peak_added_mb"]); cols.append(c)
x = np.arange(len(names))
a1.bar(x, lat, color=cols, width=0.7); a1.errorbar(x, lat, yerr=[np.zeros(len(x)), np.array(p95) - np.array(lat)], fmt="none", ecolor=INK, capsize=2, lw=0.9)
a1.set_xticks(x); a1.set_xticklabels(names, fontsize=6.5, rotation=90); a1.set_ylabel("ms per image (median, whisker = p95)")
a2.bar(x, mem, color=cols, width=0.7); a2.set_xticks(x); a2.set_xticklabels(names, fontsize=6.5, rotation=90)
a2.set_ylabel("memory added during inference (MB)")
fig.tight_layout(); fig.savefig(OUT / "3_latency_memory.png", dpi=160); plt.close(fig)

# 4 ---------------------------------------------------------------- precision / recall / F1
fig, ax = plt.subplots(figsize=(10, 4.8)); style(ax)
rows = [("FP32", S[("ONNX", "m0_fp32")])]
for lab, b, s in D:
    rows.append((lab.split("  ")[0] + " pc", S[key("ONNX", b, "pc", s)]))
x = np.arange(len(rows)); w = 0.26
for j, (m, col) in enumerate([("precision@0.25", "#1f4e9e"), ("recall@0.25", "#d98c1f"), ("F1@0.25", "#1b7f5c")]):
    v = [r[m] for _, r in rows]
    bars = ax.bar(x + (j - 1) * w, v, w, color=col, label=m.split("@")[0].capitalize() if "F1" not in m else "F1")
    for b_, vv in zip(bars, v):
        ax.text(b_.get_x() + w / 2, vv + 0.005, f"{vv:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in rows]); ax.set_ylim(0, 0.8); ax.legend(frameon=False, ncol=3)
ax.set_ylabel("score at confidence threshold 0.25 (ONNX per-channel)")
fig.tight_layout(); fig.savefig(OUT / "4_precision_recall_f1.png", dpi=160); plt.close(fig)

# 5 ---------------------------------------------------------------- pixel vs normalised control
fig, ax = plt.subplots(figsize=(7, 4.4)); style(ax)
labs = ["per-channel", "per-tensor"]
pix = [S[("ONNX", "m1_all_int8_pc")]["mAP50-95"], S[("ONNX", "m1_all_int8_pt")]["mAP50-95"]]
nrm = [S[("ONNX", "m1_all_int8_pc_norm")]["mAP50-95"], S[("ONNX", "m1_all_int8_pt_norm")]["mAP50-95"]]
for j, (v, lab, col) in enumerate([(pix, "boxes in pixels (0-640)", "#c0392b"), (nrm, "boxes normalised (0-1)", "#1f4e9e")]):
    bars = ax.bar(np.arange(2) + (j - 0.5) * 0.35, v, 0.35, color=col, label=lab)
    for b_, vv in zip(bars, v):
        ax.text(b_.get_x() + 0.175, vv + 0.004, f"{vv:.3f}", ha="center", fontsize=10)
ax.axhline(base_map, color=GRAY, ls="--", lw=1.2); ax.text(1.5, base_map + 0.004, "FP32 baseline", color=GRAY, ha="right", fontsize=9)
ax.set_xticks(range(2)); ax.set_xticklabels(labs); ax.set_ylabel("mAP50-95 (all-INT8 ONNX model)"); ax.set_ylim(0, 0.32)
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(); fig.savefig(OUT / "5_box_scale_fix.png", dpi=160); plt.close(fig)

# 6 ---------------------------------------------------------------- layer sensitivity
sens = json.loads((R / "sensitivity.json").read_text()); b = sens["fp32"]
lay = [(int(k[1:]), (b - v) / b * 100) for k, v in sens.items() if k != "fp32"]
lay.sort()
fig, ax = plt.subplots(figsize=(11, 4.8)); style(ax)
cols = ["#1f4e9e" if i < 10 else "#d98c1f" if i < 24 else "#c0392b" for i, _ in lay]
qd = json.loads((R / "sensitivity_qdq.json").read_text())
bars_ = ax.bar([i for i, _ in lay], [v for _, v in lay], color=cols, width=0.75)
for (i, _), b_ in zip(lay, bars_):
    if qd[f"L{i:02d}"] == 0:                       # layer got no Q/DQ when isolated: not a real measurement
        b_.set_facecolor("white"); b_.set_edgecolor("#999999"); b_.set_hatch("///")
        ax.text(i, 0.25, "not quantized", ha="center", fontsize=7, color="#777777", rotation=90, va="bottom")
ax.axhspan(-0.3, 0.3, color="#f1f1f1", zorder=0)
ax.set_xticks(range(25)); ax.set_xlabel("YOLOv5 layer index"); ax.set_ylabel("accuracy lost when ONLY this layer is INT8 (%)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#1f4e9e", "#d98c1f", "#c0392b")], labels=["backbone (0-9)", "neck / PANet (10-23)", "Detect head (24)"], frameon=False, loc="upper left")
ax.annotate("Detect head", (24, lay[-1][1]), xytext=(21, lay[-1][1] - 0.4), arrowprops=dict(arrowstyle="->", color="#c0392b"), color="#c0392b")
fig.tight_layout(); fig.savefig(OUT / "6_layer_sensitivity.png", dpi=160); plt.close(fig)

# 7 ---------------------------------------------------------------- calibration size
cs_path = R / "calib_size.json"
if cs_path.exists():
    cs = json.loads(cs_path.read_text())
    cs["500"] = {"mAP50-95": S[("ONNX", "m3_backbone_neck_int8_pc")]["mAP50-95"]}
    ns = sorted(int(k) for k in cs)
    fig, ax = plt.subplots(figsize=(7.5, 4.4)); style(ax)
    ax.plot(ns, [cs[str(n)]["mAP50-95"] for n in ns], marker="o", color="#1f4e9e", lw=2)
    ax.axhline(base_map, color=GRAY, ls="--", lw=1.2); ax.text(ns[-1], base_map + 0.0006, "FP32", color=GRAY, ha="right")
    for n in ns:
        ax.text(n, cs[str(n)]["mAP50-95"] - 0.0022, f"{cs[str(n)]['mAP50-95']:.4f}", ha="center", fontsize=9)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.minorticks_off()
    ax.set_xlabel("calibration images (train2017)"); ax.set_ylabel("mAP50-95 (M3 per-channel, ONNX)")
    fig.tight_layout(); fig.savefig(OUT / "7_calibration_size.png", dpi=160); plt.close(fig)
print("charts:", sorted(p.name for p in OUT.glob("*.png")))
