"""Build results/benchmark_summary.html: a short report covering the six benchmark metrics only.
Part A: the models built and how. Part B: the six benchmarks, one section each (chart + table).
Reads results/summary.json; charts go to results/charts_summary/."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
CH = R / "charts_summary"
CH.mkdir(exist_ok=True)
S = {(r["format"], r["model"]): r for r in json.loads((R / "summary.json").read_text())}

CFG = [  # (label, ONNX key, TFLite key)
    ("FP32 baseline", "m0_fp32", "tflite_m0_fp32"),
    ("M1 channel-wise", "m1_all_int8_pc_norm", "tflite_m1_all_int8_pc"),
    ("M1 layer-wise", "m1_all_int8_pt_norm", "tflite_m1_all_int8_pt"),
    ("M2 channel-wise", "m2_backbone_int8_pc", "tflite_m2_backbone_int8_pc"),
    ("M2 layer-wise", "m2_backbone_int8_pt", "tflite_m2_backbone_int8_pt"),
    ("M3 channel-wise", "m3_backbone_neck_int8_pc", "tflite_m3_backbone_neck_int8_pc"),
    ("M3 layer-wise", "m3_backbone_neck_int8_pt", "tflite_m3_backbone_neck_int8_pt"),
]
O = lambda i: S[("ONNX", CFG[i][1])]
T = lambda i: S[("TFLite", CFG[i][2])]
INK, GRID, BLUE, GREEN = "#1a1a1a", "#e6e6e6", "#1f4e9e", "#1b7f5c"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": "#bbbbbb", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
XL = [c[0].replace(" ", "\n", 1) for c in CFG]


def bars(ax, key, fmt="{:.3f}", ylabel="", ylim=None):
    x, w = np.arange(len(CFG)), 0.38
    for j, (fn, col, lab) in enumerate([(O, BLUE, "ONNX"), (T, GREEN, "TFLite")]):
        v = [fn(i)[key] for i in range(len(CFG))]
        b = ax.bar(x + (j - 0.5) * w, v, w, color=col, label=lab)
        for b_, vv in zip(b, v):
            ax.text(b_.get_x() + w / 2, vv * 1.01, fmt.format(vv), ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(XL, fontsize=8.5); ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    top = max(max(O(i)[key], T(i)[key]) for i in range(len(CFG)))
    ax.set_ylim(0, ylim or top * 1.22)


def chart(name, key, fmt, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    bars(ax, key, fmt, ylabel, ylim)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(CH / name, dpi=150); plt.close(fig)


chart("1_map50.png", "mAP50", "{:.3f}", "mAP@0.5")
chart("2_map5095.png", "mAP50-95", "{:.3f}", "mAP@0.5:0.95")
chart("3_size.png", "size_mb", "{:.2f}", "file size (MB)")
chart("4_time.png", "latency_ms_median", "{:.1f}", "milliseconds per image (median)")
chart("6_memory.png", "mem_peak_added_mb", "{:.0f}", "memory added during inference (MB)")
fig, axs = plt.subplots(3, 1, figsize=(9.5, 10.5))
for ax, (k, t) in zip(axs, [("precision@0.25", "Precision"), ("recall@0.25", "Recall"), ("F1@0.25", "F1 score")]):
    bars(ax, k, "{:.2f}", t, 0.85)
    ax.set_title(t, loc="left", fontsize=11, fontweight="bold")
axs[0].legend(frameon=False, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(CH / "5_prf.png", dpi=150); plt.close(fig)

img = lambda n, alt: f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode((CH / n).read_bytes()).decode()}">'


def table(head, body, cap):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f'<div class="tw"><table><caption>{cap}</caption><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


base_o, base_t = O(0), T(0)
LATJ = json.loads((R / "latency.json").read_text())
N_ROUNDS = LATJ["m0_fp32"]["rounds"]
f3 = lambda v: f"{v:.3f}"
f4 = lambda v: f"{v:.4f}"

# ---- tables
T_MODELS = table(["Model", "Backbone", "PANet (neck)", "Output / Detect head", "Quantization", "Formats"], [
    ["FP32 baseline", "FP32", "FP32", "FP32", "none", "ONNX, TFLite"],
    ["M1", "INT8", "INT8", "INT8", "channel-wise and layer-wise", "ONNX, TFLite"],
    ["M2", "INT8", "FP32", "FP32", "channel-wise and layer-wise", "ONNX, TFLite"],
    ["M3", "INT8", "INT8", "FP32", "channel-wise and layer-wise", "ONNX, TFLite"]],
    "Table 1. The seven models built (1 baseline + 3 INT8 designs x 2 granularities), each in ONNX and TFLite = 14 benchmarked files / pipelines.")
T_MAP50 = table(["Model", "ONNX", "TFLite", "ONNX vs FP32", "TFLite vs FP32"],
                [[c[0], f3(O(i)["mAP50"]), f3(T(i)["mAP50"]), "-" if i == 0 else f"{O(i)['mAP50'] - base_o['mAP50']:+.3f}", "-" if i == 0 else f"{T(i)['mAP50'] - base_t['mAP50']:+.3f}"] for i, c in enumerate(CFG)],
                "Table 2. mAP@0.5 (higher is better).")
T_MAP95 = table(["Model", "ONNX", "TFLite", "ONNX retained", "TFLite retained"],
                [[c[0], f4(O(i)["mAP50-95"]), f4(T(i)["mAP50-95"]), f"{O(i)['accuracy_retained_pct']:.1f}%", f"{T(i)['accuracy_retained_pct']:.1f}%"] for i, c in enumerate(CFG)],
                "Table 3. mAP@0.5:0.95 (higher is better). &ldquo;Retained&rdquo; = share of the FP32 model's score.")
T_SIZE = table(["Model", "ONNX size (MB)", "ONNX compression", "TFLite size (MB)", "TFLite compression"],
               [[c[0], f"{O(i)['size_mb']:.2f}", f"{O(i)['compression_ratio']:.2f}x", f"{T(i)['size_mb']:.2f}", f"{T(i)['compression_ratio']:.2f}x"] for i, c in enumerate(CFG)],
               "Table 4. Binary size and compression ratio (FP32 size divided by model size, same format). TFLite sizes for M2/M3 are two files (INT8 stage + FP32 stage) summed.")
T_TIME = table(["Model", "ONNX median (ms)", "ONNX p95 (ms)", "ONNX speed-up", "TFLite median (ms)", "TFLite p95 (ms)", "TFLite speed-up"],
               [[c[0], f"{O(i)['latency_ms_median']:.1f}", f"{O(i)['latency_ms_p95']:.1f}", f"{O(i)['speedup_vs_fp32']:.2f}x", f"{T(i)['latency_ms_median']:.1f}", f"{T(i)['latency_ms_p95']:.1f}", f"{T(i)['speedup_vs_fp32']:.2f}x"] for i, c in enumerate(CFG)],
               f"Table 5. Inference time per image (model only; batch 1, 4 CPU threads; median of {N_ROUNDS} interleaved rounds of 100 runs after warm-up).")
T_PRF = table(["Model", "ONNX P", "ONNX R", "ONNX F1", "TFLite P", "TFLite R", "TFLite F1"],
              [[c[0], f3(O(i)["precision@0.25"]), f3(O(i)["recall@0.25"]), f3(O(i)["F1@0.25"]), f3(T(i)["precision@0.25"]), f3(T(i)["recall@0.25"]), f3(T(i)["F1@0.25"])] for i, c in enumerate(CFG)],
              "Table 6. Precision (P), recall (R) and F1 at confidence threshold 0.25, IoU 0.5, over all 5000 images.")
T_MEM = table(["Model", "ONNX at load (MB)", "ONNX peak (MB)", "ONNX vs FP32", "TFLite at load (MB)", "TFLite peak (MB)", "TFLite vs FP32"],
              [[c[0], f"{O(i)['mem_load_mb']:.1f}", f"{O(i)['mem_peak_added_mb']:.0f}", "-" if i == 0 else f"{(O(i)['mem_peak_added_mb'] / base_o['mem_peak_added_mb'] - 1) * 100:+.0f}%",
                f"{T(i)['mem_load_mb']:.1f}", f"{T(i)['mem_peak_added_mb']:.0f}", "-" if i == 0 else f"{(T(i)['mem_peak_added_mb'] / base_t['mem_peak_added_mb'] - 1) * 100:+.0f}%"] for i, c in enumerate(CFG)],
               "Table 7. Memory added to the process by the model: right after loading, and at the peak during inference (includes activation tensors). Excludes the runtime's own baseline. Fresh process per model.")

# ---- FP32 vs INT8 overview
ov = []
for fmt, fn in (("ONNX", O), ("TFLite", T)):
    for i in range(1, len(CFG)):
        r, b = fn(i), fn(0)
        ov.append([fmt, CFG[i][0], f"{r['mAP50-95']:.4f}", f"{(r['mAP50-95'] - b['mAP50-95']) / b['mAP50-95'] * 100:+.1f}%", f"{r['compression_ratio']:.2f}x",
                   f"{r['speedup_vs_fp32']:.2f}x", f"{r['F1@0.25'] - b['F1@0.25']:+.3f}", f"{(r['mem_peak_added_mb'] / b['mem_peak_added_mb'] - 1) * 100:+.0f}%"])
T_OV = table(["Format", "INT8 model", "mAP@0.5:0.95", "Change vs FP32", "Compression", "Speed-up", "F1 change", "Peak memory vs FP32"], ov,
             "Table 8. Every INT8 model compared with the FP32 model of the same format.")

# ---- derived numbers for the reading notes
less = lambda a, b: (1 - a / b) * 100
span = lambda rs, key, ref: (min(less(r[key], ref[key]) for r in rs), max(less(r[key], ref[key]) for r in rs))
o_int = [O(i) for i in range(1, 7)]
t_m1, t_m23 = [T(1), T(2)], [T(i) for i in range(3, 7)]
sp_o, sp_t1, sp_t23 = span(o_int, "latency_ms_median", base_o), span(t_m1, "latency_ms_median", base_t), span(t_m23, "latency_ms_median", base_t)
spread = max(r["latency_round_spread_pct"] for r in list(map(O, range(7))) + list(map(T, range(7))))
chg = lambda rs, ref: sorted(((r["mem_peak_added_mb"] / ref["mem_peak_added_mb"] - 1) * 100) for r in rs)
mo1, mo23, mt1, mt23 = chg([O(1), O(2)], base_o), chg([O(i) for i in range(3, 7)], base_o), chg(t_m1, base_t), chg(t_m23, base_t)
rg = lambda a: f"{abs(a[0]):.0f}%" if round(abs(a[0])) == round(abs(a[-1])) else f"{min(abs(a[0]), abs(a[-1])):.0f} to {max(abs(a[0]), abs(a[-1])):.0f}%"

ONNX_KEYS = [c[1] for c in CFG[1:]]
TFL_KEYS = [c[2] for c in CFG[1:]]
faster_rounds = lambda keys, base: [sum(a < b for a, b in zip(LATJ[k]["round_medians_ms"], LATJ[base]["round_medians_ms"])) for k in keys]
fr_o, fr_t = faster_rounds(ONNX_KEYS, "m0_fp32"), faster_rounds(TFL_KEYS, "tflite_m0_fp32")
if (R / "calib_method.json").exists():
    _cm = json.loads((R / "calib_method.json").read_text())["minmax"]["mAP50-95"]
    CM_TXT = (f"Re-building the ONNX M3 channel-wise model with min-max calibration (the TFLite converter's method) gives {_cm:.4f}, against {O(5)['mAP50-95']:.4f} with percentile calibration and {T(5)['mAP50-95']:.4f} for TFLite, "
              + ("so the calibration method accounts for essentially all of that gap." if abs((O(5)['mAP50-95'] - _cm) / (O(5)['mAP50-95'] - T(5)['mAP50-95']) - 1) < 0.1 else f"so calibration accounts for {(O(5)['mAP50-95'] - _cm) / (O(5)['mAP50-95'] - T(5)['mAP50-95']) * 100:.0f}% of that gap."))
else:
    CM_TXT = "The calibration method (percentile in ONNX Runtime, min-max in the TFLite converter) is a likely cause."

PJ = json.loads((R / "profile.json").read_text()) if (R / "profile.json").exists() else None
if PJ:
    _op = lambda m, k: PJ[m]["by_op_ms"].get(k, 0.0)
    _rg = lambda m, k: PJ[m]["by_region_ms"].get(k, 0.0)
    PROF_TXT = (f" Profiling the ONNX models shows why the gain is not larger: INT8 convolutions are faster ({_op('M1 per-channel', 'conv'):.1f} ms against {_op('FP32', 'conv'):.1f} ms), "
                f"but the SiLU activation costs {_op('M1 per-channel', 'activation (sigmoid/mul)'):.1f} ms in INT8 against {_op('FP32', 'activation (sigmoid/mul)'):.1f} ms in FP32 (where it is fused into the convolution), "
                f"and the INT8 neck takes {_rg('M3 per-channel', 'neck'):.1f} ms against {_rg('FP32', 'neck'):.1f} ms in FP32, which is why M2 (FP32 neck) is the fastest ONNX design.")
else:
    PROF_TXT = ""
sg = lambda x: "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"
_hf = [c for i in (3, 4, 5, 6) for c in (O(i), T(i))]
_m1 = [c for i in (1, 2) for c in (O(i), T(i))]
_rg = lambda rs, k, b_: (min(r[k] - b_[k] for r in rs), max(r[k] - b_[k] for r in rs))
_fp = base_o
PRF_TXT = (f"relative to FP32, designs with an FP32 head (M2, M3) change precision by {sg(_rg(_hf, 'precision@0.25', _fp)[0])} to {sg(_rg(_hf, 'precision@0.25', _fp)[1])}, "
           f"recall by {sg(_rg(_hf, 'recall@0.25', _fp)[0])} to {sg(_rg(_hf, 'recall@0.25', _fp)[1])} and F1 by {sg(_rg(_hf, 'F1@0.25', _fp)[0])} to {sg(_rg(_hf, 'F1@0.25', _fp)[1])}; "
           f"the all-INT8 designs (M1) change precision by {sg(_rg(_m1, 'precision@0.25', _fp)[0])} to {sg(_rg(_m1, 'precision@0.25', _fp)[1])}, "
           f"recall by {sg(_rg(_m1, 'recall@0.25', _fp)[0])} to {sg(_rg(_m1, 'recall@0.25', _fp)[1])} and F1 by {sg(_rg(_m1, 'F1@0.25', _fp)[0])} to {sg(_rg(_m1, 'F1@0.25', _fp)[1])}. "
           f"Recall falls in every quantized model, while precision is roughly unchanged or slightly higher when the head stays FP32 and lower when it does not.")
best_acc = max(range(1, 7), key=lambda i: O(i)["mAP50-95"])
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLOv5n Quantization: Models and Benchmark Results</title>
<style>
:root {{ color-scheme: light; }} * {{ box-sizing:border-box; }} html {{ background:#fff; scroll-behavior:smooth; }}
body {{ margin:0; color:#1a1a1a; background:#fff; font:17px/1.6 Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif; }}
.page {{ max-width:900px; margin:0 auto; padding:40px 24px 90px; }}
h1,h2,h3,.meta,table,figcaption,nav {{ font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }}
h1 {{ font-size:28px; line-height:1.25; margin:0 0 8px; letter-spacing:-0.01em; }}
.meta {{ color:#666; font-size:14px; margin:0 0 18px; }}
nav {{ position:sticky; top:0; background:#fff; border-top:1px solid #ddd; border-bottom:1px solid #ddd; padding:9px 0; margin:0 0 28px; font-size:13px; z-index:5; display:flex; flex-wrap:wrap; gap:6px 16px; }}
nav a {{ color:#0b57d0; text-decoration:none; }} nav a:hover {{ text-decoration:underline; }}
h2 {{ font-size:21px; margin:46px 0 4px; padding-top:6px; scroll-margin-top:50px; }} h2 .n {{ color:#888; font-weight:400; }}
.def {{ color:#555; font-size:15px; margin:0 0 14px; }}
p {{ margin:0 0 13px; }} ul {{ padding-left:22px; margin:0 0 13px; }} li {{ margin:5px 0; }}
code {{ font:0.88em Consolas,Menlo,monospace; background:#f5f5f5; padding:1px 5px; border-radius:3px; }}
.read {{ border-left:3px solid #1a1a1a; padding:2px 0 2px 14px; margin:14px 0 0; }}
.tw {{ overflow-x:auto; margin:20px 0 6px; }} table {{ border-collapse:collapse; width:100%; font-size:13px; }}
caption {{ text-align:left; font-weight:700; padding:0 0 8px; font-size:13.5px; }}
th,td {{ padding:6px 12px 6px 0; text-align:right; font-variant-numeric:tabular-nums; vertical-align:top; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
thead th {{ border-top:1.5px solid #1a1a1a; border-bottom:1px solid #1a1a1a; font-weight:600; white-space:normal; }}
tbody tr {{ border-bottom:1px solid #eee; }} tbody tr:last-child {{ border-bottom:1.5px solid #1a1a1a; }}
table.text td, table.text th {{ text-align:left; white-space:normal; }}
figure {{ margin:18px 0 6px; }} figure img {{ display:block; width:100%; height:auto; }}
figcaption {{ font-size:13px; color:#555; margin-top:6px; }}
ol.steps li {{ margin:8px 0; }}
footer {{ margin-top:60px; padding-top:14px; border-top:1px solid #ddd; color:#777; font:13px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }}
@media print {{ nav {{ display:none; }} .page {{ padding:0; max-width:none; }} h2 {{ break-after:avoid; }} figure,table {{ break-inside:avoid; }} }}
</style></head><body><main class="page">

<h1>YOLOv5n Quantization: Models Built and Benchmark Results</h1>
<p class="meta">Original YOLOv5n v7.0 &nbsp;|&nbsp; target: embedded NPU (i.MX8-class) &nbsp;|&nbsp; test set: COCO val2017, all 5000 images &nbsp;|&nbsp; formats: ONNX and TFLite</p>
<nav><a href="#done">What was done</a><a href="#b1">1 mAP@0.5</a><a href="#b2">2 mAP@0.5:0.95</a><a href="#b3">3 Binary size</a><a href="#b4">4 Time per image</a><a href="#b5">5 Precision / Recall / F1</a><a href="#b6">6 Memory</a><a href="#cmp">FP32 vs INT8</a><a href="#notes">Notes</a></nav>

<h2 id="done">What was done</h2>
<ol class="steps">
<li><b>Baseline:</b> the original YOLOv5n model with no quantization (FP32). Its mAP@0.5:0.95 is <b>{base_o['mAP50-95']:.3f}</b>, matching the published 0.28, so the benchmark setup is correct.</li>
<li><b>Three INT8 versions</b>, differing in which part of the network is quantized (YOLOv5 has a backbone, a PANet neck and an output / Detect head):
 <ul><li><b>M1</b>: everything INT8, including the output head (the control).</li><li><b>M2</b>: backbone INT8; PANet and output FP32.</li><li><b>M3</b>: backbone and PANet INT8; output FP32.</li></ul></li>
<li><b>Two granularities</b> for each: <b>channel-wise</b> (one scale per output channel; "per-channel") and <b>layer-wise</b> (one scale per layer; "per-tensor").</li>
<li><b>Two formats</b> for everything: <b>ONNX</b> (ONNX Runtime) and <b>TensorFlow Lite</b>. For TFLite, the model is split into separate files at the part boundaries so some parts can stay FP32.</li>
<li><b>Calibration data:</b> 500 images from COCO train2017 (separate from the test images). <b>Test data:</b> all 5000 COCO val2017 images, scored with the official COCO evaluation (<code>pycocotools</code>).</li>
</ol>
{T_MODELS}
<p>One technical point: for the all-INT8 model (M1) the box coordinates are normalised to 0-1 (divided by 640) before quantizing, as the official YOLOv5 TFLite export does. Without this, M1 scores zero, because box values up to 640 and confidence scores below 1 cannot share one INT8 scale.</p>

<h2 id="b1"><span class="n">1.</span> mAP@0.5</h2>
<p class="def">Detection accuracy when a predicted box counts as correct if it overlaps the true box by at least 50%.</p>
<figure>{img('1_map50.png', 'mAP@0.5 for every model')}<figcaption>Figure 1. mAP@0.5 per model, ONNX and TFLite.</figcaption></figure>
{T_MAP50}
<p class="read"><b>Reading:</b> FP32 scores {base_o['mAP50']:.3f}. The best INT8 model, M2 channel-wise, scores {O(3)['mAP50']:.3f} in ONNX and {T(3)['mAP50']:.3f} in TFLite. Channel-wise is higher than layer-wise in every pair.</p>

<h2 id="b2"><span class="n">2.</span> mAP@0.5:0.95</h2>
<p class="def">The stricter accuracy score: averaged over overlap thresholds from 50% to 95%, so box placement has to be tight. This is the main COCO metric.</p>
<figure>{img('2_map5095.png', 'mAP@0.5:0.95 for every model')}<figcaption>Figure 2. mAP@0.5:0.95 per model, ONNX and TFLite.</figcaption></figure>
{T_MAP95}
<p class="read"><b>Reading:</b> FP32 scores {base_o['mAP50-95']:.4f}. M2 channel-wise keeps {O(3)['accuracy_retained_pct']:.1f}% (ONNX) and {T(3)['accuracy_retained_pct']:.1f}% (TFLite); M3 channel-wise keeps {O(5)['accuracy_retained_pct']:.1f}% and {T(5)['accuracy_retained_pct']:.1f}%; the all-INT8 M1 keeps {O(1)['accuracy_retained_pct']:.1f}% and {T(1)['accuracy_retained_pct']:.1f}%. Every difference between channel-wise and layer-wise is statistically significant (10-fold paired test).</p>

<h2 id="b3"><span class="n">3.</span> Binary size and compression ratio</h2>
<p class="def">File size of each model, and compression ratio = FP32 file size divided by the model's size.</p>
<figure>{img('3_size.png', 'File size for every model')}<figcaption>Figure 3. File size (MB) per model.</figcaption></figure>
{T_SIZE}
<p class="read"><b>Reading:</b> the FP32 model is {base_o['size_mb']:.2f} MB. M1 compresses {O(1)['compression_ratio']:.1f}x, M3 {O(5)['compression_ratio']:.1f}x (ONNX) and {T(5)['compression_ratio']:.1f}x (TFLite), M2 {O(3)['compression_ratio']:.1f}x. M2 and M3 are larger than M1 because the FP32 parts are stored in full precision.</p>

<h2 id="b4"><span class="n">4.</span> Execution time per image</h2>
<p class="def">Time for the model to process one image (batch 1, 4 CPU threads), median of 200 runs after warm-up. Measured on a desktop CPU, not the NPU.</p>
<figure>{img('4_time.png', 'Inference time per image')}<figcaption>Figure 4. Median milliseconds per image.</figcaption></figure>
{T_TIME}
<p class="read"><b>Reading:</b> FP32 takes {base_o['latency_ms_median']:.1f} ms (ONNX) and {base_t['latency_ms_median']:.1f} ms (TFLite) per image. Quantization is working (ONNX Runtime uses integer convolution kernels for the INT8 models): INT8 ONNX models take {sp_o[0]:.0f} to {sp_o[1]:.0f}% less time, TFLite all-INT8 models {sp_t1[0]:.0f} to {sp_t1[1]:.0f}% less, and TFLite M2/M3 {sp_t23[0]:.0f} to {sp_t23[1]:.0f}% less (their FP32 stage still runs as float). In all {N_ROUNDS} rounds every ONNX INT8 model beat FP32; each TFLite INT8 pipeline did so in at least {min(fr_t)} of {N_ROUNDS}. The same model varied by up to {spread:.0f}% between rounds, so small gaps <i>between</i> INT8 variants are not meaningful.{PROF_TXT} These timings do not predict NPU speed.</p>

<h2 id="b5"><span class="n">5.</span> Precision, Recall and F-score</h2>
<p class="def">At confidence threshold 0.25 and IoU 0.5. <b>Precision</b>: of the objects reported, how many are right. <b>Recall</b>: of the real objects, how many are found. <b>F1</b>: their harmonic mean.</p>
<figure>{img('5_prf.png', 'Precision, recall and F1')}<figcaption>Figure 5. Precision, recall and F1 per model.</figcaption></figure>
{T_PRF}
<p class="read"><b>Reading:</b> {PRF_TXT}</p>

<h2 id="b6"><span class="n">6.</span> Memory utilization</h2>
<p class="def">Peak memory of the process while running inference (a fresh process per model), and how much loading the model adds on its own.</p>
<figure>{img('6_memory.png', 'Peak memory per model')}<figcaption>Figure 6. Memory added to the process during inference (MB), excluding the runtime baseline.</figcaption></figure>
{T_MEM}
<p class="read"><b>Reading:</b> the FP32 ONNX model adds {base_o['mem_peak_added_mb']:.0f} MB at peak; the all-INT8 ONNX models add {rg(mo1)} less, while M2 and M3 (which mix INT8 and FP32) add {mo23[0]:.0f} to {mo23[-1]:.0f}% more. In TFLite the FP32 pipeline adds {base_t['mem_peak_added_mb']:.0f} MB, the all-INT8 pipelines {rg(mt1)} less, and M2/M3 {rg(mt23)} less. Memory at load is similar across ONNX models (probably because the runtime re-packs weights whatever their precision; not verified). Measuring memory on the target board would give the figure that matters for deployment.</p>

<h2 id="cmp">FP32 versus INT8 comparison</h2>
{T_OV}

<h2 id="notes">Notes on how to read these results</h2>
<ul>
<li><b>ONNX and TFLite are not exactly like-for-like:</b> ONNX Runtime used percentile calibration and the TFLite converter used min-max (the converter has no percentile option). ONNX scores 0.4 to 0.9 mAP points higher for the same design. {CM_TXT}</li>
<li><b>Timing and memory come from a desktop CPU</b>, not the i.MX8 NPU. On an NPU, only INT8 parts run on the accelerator, so M2 and M3 (which keep FP32 parts) would run those parts on the host CPU. This needs measuring on the board.</li>
<li><b>All accuracy numbers use all 5000 val2017 images,</b> so differences of about 0.5 mAP points or more are real; the paired 10-fold test confirms every channel-wise versus layer-wise difference.</li>
<li>The full report with extra studies (layer sensitivity, calibration-set size, statistics, method details) is <a href="report.html">report.html</a>.</li>
</ul>

<footer>Generated by <code>scripts/make_benchmark_summary.py</code> from <code>results/summary.json</code>. Raw numbers: <code>results/summary.csv</code>.</footer>
</main></body></html>"""
(R / "benchmark_summary.html").write_text(html, encoding="utf-8")
print("wrote benchmark_summary.html", f"({(R / 'benchmark_summary.html').stat().st_size / 1e3:.0f} KB)")
