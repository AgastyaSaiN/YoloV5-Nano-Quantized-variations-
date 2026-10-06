"""Turn results/benchmark_results.json into three beginner-friendly charts in results/charts/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results" / "benchmark_results.json").read_text())
NOTITLE = "--no-titles" in sys.argv   # clean versions for report.html (captions live in the HTML)
OUT = ROOT / "results" / ("charts_for_report" if NOTITLE else "charts")
OUT.mkdir(parents=True, exist_ok=True)

LABELS = {
    "fp32_reference": "Original\n(FP32)",
    "v1_percentile_head": "v1\nPercentile",
    "v3_percentile9999_head": "v3\nPercentile\n99.99",
    "v4_percentile_head_neck": "v4\n+ extra FP32\nlayers",
    "v2_minmax_head": "v2\nMin-Max",
    "ctrl_percentile_all": "Control\n(head also INT8)",
}
ORDER = list(LABELS)
GRAY, BLUE, RED, INK, GRID = "#8a8f98", "#2f6fdb", "#d64545", "#1f2430", "#e6e8ec"
color = lambda k: GRAY if k == "fp32_reference" else RED if k.startswith("ctrl") else BLUE

plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

def set_title(ax, text, **kw):
    if not NOTITLE:
        ax.set_title(text, **kw)


def style(ax):
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)

# 1) Accuracy
fig, ax = plt.subplots(figsize=(10, 5)); style(ax)
vals = [R[k]["mAP50-95"] for k in ORDER]
bars = ax.bar([LABELS[k] for k in ORDER], vals, color=[color(k) for k in ORDER], width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontweight="bold")
ax.annotate("Quantizing the head too\ndestroys the detections", xy=(5, 0.025), xytext=(5, 0.13), ha="center",
            color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.set_ylabel("Accuracy score (mAP50-95, higher = better)"); ax.set_ylim(0, 0.40)
set_title(ax, "1. Accuracy: all head-protected versions stay within ~1% of the original", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "1_accuracy.png", dpi=160); plt.close(fig)

# 2) Size and speed relative to the original (control excluded: it is broken)
keys = [k for k in ORDER if not k.startswith("ctrl")]
base = R["fp32_reference"]
fig, ax = plt.subplots(figsize=(10, 5)); style(ax)
x = range(len(keys)); w = 0.38
size = [R[k]["size_mb"] / base["size_mb"] * 100 for k in keys]
lat = [R[k]["latency_ms_median"] / base["latency_ms_median"] * 100 for k in keys]
b1 = ax.bar([i - w / 2 for i in x], size, w, color="#2f6fdb", label="File size")
b2 = ax.bar([i + w / 2 for i in x], lat, w, color="#59b38a", label="Time per image")
for bars_ in (b1, b2):
    for b in bars_:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%", ha="center", fontsize=10)
ax.axhline(100, color=GRAY, ls="--", lw=1); ax.text(len(keys) - 0.5, 102, "original = 100%", color=GRAY, ha="right")
ax.set_xticks(list(x)); ax.set_xticklabels([LABELS[k] for k in keys]); ax.set_ylim(0, 120)
ax.set_ylabel("% of the original (lower = better)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
set_title(ax, "2. Size and speed: about half the size, 10-20% faster", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "2_size_and_speed.png", dpi=160); plt.close(fig)

# 3) Accuracy lost, zoomed
keys = [k for k in ORDER if k not in ("fp32_reference", "ctrl_percentile_all")]
drops = [(base["mAP50-95"] - R[k]["mAP50-95"]) / base["mAP50-95"] * 100 for k in keys]
fig, ax = plt.subplots(figsize=(10, 4.6)); style(ax)
bars = ax.bar([LABELS[k] for k in keys], drops, color=BLUE, width=0.55)
for b, v in zip(bars, drops):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"-{v:.1f}%", ha="center", fontweight="bold")
ax.axhspan(0, 0.5, color="#f3f4f6", zorder=0)
ax.text(-0.45, 1.65, "Grey band: differences below ~0.5% are within noise (only 1000 test images)", ha="left", color=GRAY, fontsize=9.5)
ax.set_ylabel("Accuracy lost vs original (%, lower = better)"); ax.set_ylim(0, 1.8)
set_title(ax, "3. Zoom: accuracy lost by each protected version", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "3_accuracy_lost_zoomed.png", dpi=160); plt.close(fig)
print("charts written to", OUT)
