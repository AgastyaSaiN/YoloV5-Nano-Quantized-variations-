"""Results by period: Day 2 implementations, Day 3 implementations and the current stage, with the same benchmarks charted side by side
(older periods in lighter shades). Shared by report.html and benchmark_summary.html.
Use eras_html(img_fn), era_chart_files(out_dir) and era_figure(metric, img_fn)."""
import html
from pathlib import Path

import extra_sections as E
from extra_sections import BASE, acc, mem

prf = E.J("prf.json")
LAT = dict(E.lat)
# The Day 2 bars keep the Day 2 timing as first measured (results/logs/latency_before_current_stage.json); the later re-measurement of the same models
# (taken in the session that measured the current-stage ONNX/TFLite models) differs by a few percent and is used in the main tables.
_d2 = Path(__file__).resolve().parent.parent / "results" / "logs" / "latency_before_current_stage.json"
if _d2.exists():
    import json as _json
    _old = _json.loads(_d2.read_text())
    for _k in KEYS_DAY2 if False else ("m0_fp32", "m1_all_int8_pc_norm", "m2_backbone_int8_pc", "m3_backbone_neck_int8_pc"):
        if _k in _old:
            LAT[_k] = _old[_k]
SHADES = {"Day 2": "#bcd0ee", "Day 3": "#6f98d6", "Current": "#1f4e9e"}
DESIGNS = [("M1 all INT8", "m1"), ("M2 backbone INT8", "m2"), ("M3 backbone + neck INT8", "m3")]
KEYS = {  # period -> design -> result key (accuracy / latency / memory / prf all share the key)
    "Day 2": {"m1": "m1_all_int8_pc_norm", "m2": "m2_backbone_int8_pc", "m3": "m3_backbone_neck_int8_pc"},
    "Day 3": {"m1": "ovq_m1_mixed", "m2": "ovq_m2_mixed", "m3": "ovq_m3_mixed"},
    "Current": {"m1": "ovq_f512_m1_mixed", "m2": "ovq_f512_m2_mixed", "m3": "ovq_f512_m3_mixed"},
}
FP32 = {"Day 2": "m0_fp32", "Day 3": "ov_m0_fp32"}                     # FP32 baseline in each period's runtime
RUNTIME = {"Day 2": "ONNX Runtime", "Day 3": "OpenVINO", "Current": "OpenVINO"}
ACCKEY = {"ov_m0_fp32": "m0_fp32"}


def value(metric, key):
    k = ACCKEY.get(key, key)
    if metric == "map50":
        return acc[k]["mAP50"]
    if metric == "map":
        return acc[k]["mAP50-95"]
    if metric == "size":
        return acc[k]["size_mb"]
    if metric == "time":
        return LAT[key]["latency_ms_median"]
    if metric == "f1":
        return prf[k]["F1@0.25"]
    if metric == "mem":
        return mem[key]["peak_mb"]
    raise KeyError(metric)


METRICS = {"map50": ("mAP@0.5", "{:.3f}"), "map": ("mAP@0.5:0.95", "{:.3f}"), "size": ("file size (MB)", "{:.2f}"),
           "time": ("model time per image (ms, median)", "{:.1f}"), "f1": ("F1 score at confidence 0.25", "{:.3f}"), "mem": ("memory added during inference (MB)", "{:.0f}")}


def _draw(ax, metric):
    import numpy as np
    label, fmt = METRICS[metric]
    groups = [("FP32\n640 px", None)] + [(d[0].replace(" ", "\n", 1), d[1]) for d in DESIGNS]
    periods = list(SHADES)
    w = 0.26
    for gi, (_, dk) in enumerate(groups):
        for pi, per in enumerate(periods):
            if dk is None:
                key = FP32.get(per)
                if key is None or (metric == "mem" and key not in mem):
                    continue
            else:
                key = KEYS[per][dk]
            try:
                v = value(metric, key)
            except KeyError:
                continue
            x = gi + (pi - 1) * w
            ax.bar(x, v, w * 0.95, color=SHADES[per], label=per if gi == 0 or (gi == 1 and per == "Current") else None)
            ax.text(x, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel(label); ax.yaxis.grid(True, color="#e6e6e6"); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    top = ax.get_ylim()[1]; ax.set_ylim(0, top * 1.18)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend([seen[p] for p in SHADES if p in seen], [p for p in SHADES if p in seen], frameon=False, ncol=3, loc="upper right", fontsize=9)


def era_chart_files(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    names = {}
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(9.5, 4.0))
        _draw(ax, m)
        fig.tight_layout(); fig.savefig(out_dir / f"era_{m}.png", dpi=150); plt.close(fig)
        names[m] = f"era_{m}.png"
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.2))
    for ax, m in zip(axs.flat, METRICS):
        _draw(ax, m)
    fig.tight_layout(); fig.savefig(out_dir / "era_all.png", dpi=140); plt.close(fig)
    names["all"] = "era_all.png"
    return names


def era_figure(metric, img_fn, names):
    cap = {"map50": "mAP@0.5", "map": "mAP@0.5:0.95", "size": "File size", "time": "Model time per image", "f1": "F1 score", "mem": "Memory"}[metric]
    return (f"<figure>{img_fn(names[metric], cap + ' by period')}<figcaption><b>{cap} by period.</b> The same three designs in each period; lighter bars are older. "
            f"Day 2: ONNX Runtime. Day 3: OpenVINO at 640 px. Current: OpenVINO at 512 px. The FP32 group shows the unquantized model in the Day 2 and Day 3 runtimes.</figcaption></figure>")


def _row(per, dk, label):
    k = KEYS[per][dk]
    a = acc[k]
    base_ms = LAT[FP32.get(per, "ov_m0_fp32")]["latency_ms_median"]
    return [per, RUNTIME[per], label, f"{a['mAP50-95']:.4f}", f"{a['mAP50-95'] / BASE * 100:.1f}%", f"{a['size_mb']:.2f}", f"{LAT[k]['latency_ms_median']:.1f}",
            f"{base_ms / LAT[k]['latency_ms_median']:.2f}x", f"{prf[k]['F1@0.25']:.3f}", f"{mem[k]['peak_mb']:.0f}"]


def eras_html(img_fn, names):
    rows = [_row(per, dk, lab) for dk, lab in ((d[1], d[0]) for d in DESIGNS) for per in SHADES]
    rows.sort(key=lambda r: ([d[0] for d in DESIGNS].index(r[2]), list(SHADES).index(r[0])))
    tab = E.table(["Period", "Runtime", "Design", "mAP50-95", "Retained", "Size (MB)", "Model ms", "Speed-up", "F1", "Peak memory added (MB)"], rows,
                  "The same three designs in each period (per-channel for Day 2). Speed-up is against the FP32 model of the same runtime at 640 px (Day 2: ONNX Runtime, Day 3 and Current: OpenVINO).")
    f512, f448 = KEYS["Current"]["m3"], "ovq_f448_m3_mixed"
    pp = E.PP
    return f"""<h2>Implementations by period</h2>
<p>The work is grouped into three periods. Each period is described below; the charts and tables then show the same benchmarks for all three, older periods in lighter shades. The Detect head is untouched in every model of every period.</p>

<h3>Day 2 implementations</h3>
<ul>
<li>FP32 YOLOv5n baseline and three INT8 designs: <b>M1</b> everything INT8, <b>M2</b> backbone INT8 with neck and head FP32, <b>M3</b> backbone and neck INT8 with the head FP32.</li>
<li>Each design in two granularities (per-channel and per-tensor) and two formats: ONNX (ONNX Runtime, percentile calibration) and TFLite (full-integer, min-max calibration, split into separate files so part of the network can stay FP32).</li>
<li>Box coordinates normalised to 0-1 so that the all-INT8 model works. Calibration on 500 train2017 images; testing on all 5000 val2017 images with the official COCO evaluation.</li>
<li>Extra studies: layer sensitivity, calibration-set size, 10-fold statistical comparison.</li>
</ul>

<h3>Day 3 implementations</h3>
<ul>
<li>Audit and corrections: pinned cores for timing, an idle machine, matched calibration methods for the ONNX versus TFLite comparison, and official COCO matching for precision and recall.</li>
<li>OpenVINO quantization (NNCF) of the same designs: the smallest files and a faster INT8 runtime on this CPU.</li>
<li>ReLU-activation models: R55 (12 of 57 activations replaced, 55% of the activation work) and R80 (29 of 57, 81%), fine-tuned with the Detect head frozen. They are faster but less accurate (see the tables and the speed-versus-accuracy chart).</li>
</ul>

<h3>Current stage</h3>
<ul>
<li>Reduced input size: the original network exported at 512, 448 and 416 pixels and quantized with the same OpenVINO procedure, no retraining. At 512 pixels M3 is {LAT['ov_m0_fp32']['latency_ms_median'] / LAT[f512]['latency_ms_median']:.2f}x faster than the 640 px FP32 model at {acc[f512]['mAP50-95'] / BASE * 100:.1f}% of its accuracy; at 448 pixels {LAT['ov_m0_fp32']['latency_ms_median'] / LAT[f448]['latency_ms_median']:.2f}x at {acc[f448]['mAP50-95'] / BASE * 100:.1f}%.</li>
<li>Compiled post-processing with identical output ({pp['post_ms_numpy_conf0.25']:.1f} ms down to {pp['post_ms_compiled_conf0.25']:.1f} ms per image at confidence 0.25).</li>
<li>Tried and not adopted: fine-tuning at the smaller size, with or without ReLU, scored below the un-retrained network at the same size.</li>
<li>Not yet tried: channel pruning and quantization-aware training.</li>
</ul>

<figure>{img_fn(names['all'], 'All benchmarks by period')}<figcaption><b>All six benchmarks by period.</b> Lighter bars are older periods; each group is one design.</figcaption></figure>
{tab}"""
