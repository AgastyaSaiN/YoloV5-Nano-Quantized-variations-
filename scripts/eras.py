"""The latest models in all three runtimes (ONNX Runtime, TFLite, OpenVINO): the same designs M1/M2/M3 charted side by side, plus plain-language
notes on how to read each benchmark. Shared by report.html and benchmark_summary.html.
Use eras_html(img_fn, names), era_chart_files(out_dir), era_figure(metric, img_fn, names) and era_note(metric)."""
from pathlib import Path

import extra_sections as E
from extra_sections import BASE, acc, mem

prf = E.J("prf.json")
LAT = dict(E.lat)

COLORS = {"ONNX Runtime": "#1f4e9e", "TFLite": "#1b7f5c", "OpenVINO": "#c0551b"}
DESIGNS = [("M1\nall INT8", "m1"), ("M2\nbackbone INT8", "m2"), ("M3\nbackbone+neck INT8", "m3")]
FP32 = {"ONNX Runtime": "m0_fp32", "TFLite": "tflite_m0_fp32", "OpenVINO": "ov_m0_fp32"}                         # the 640 px FP32 model in each runtime
CURRENT = {
    "ONNX Runtime": {"m1": "s512_m1_all_int8_pc", "m2": "s512_m2_backbone_int8_pc", "m3": "s512_m3_backbone_neck_int8_pc"},
    "TFLite": {"m1": "tflite_t512_m1_all_int8_pc", "m2": "tflite_t512_m2_backbone_int8_pc", "m3": "tflite_t512_m3_backbone_neck_int8_pc"},
    "OpenVINO": {"m1": "ovq_f512_m1_mixed", "m2": "ovq_f512_m2_mixed", "m3": "ovq_f512_m3_mixed"},
}
FP32_512 = {"ONNX Runtime": "f512_fp32", "TFLite": "tflite_s512_m0_fp32", "OpenVINO": "ov_f512_fp32"}          # FP32 at the same 512 px input as the INT8 models
ACCKEY = {"ov_m0_fp32": "m0_fp32", "ov_f512_fp32": "f512_fp32"}
METRICS = {"map50": ("mAP@0.5", "{:.3f}"), "map": ("mAP@0.5:0.95", "{:.3f}"), "size": ("file size (MB)", "{:.2f}"),
           "time": ("model time per image (ms, median)", "{:.1f}"), "f1": ("F1 score at confidence 0.25", "{:.3f}"), "mem": ("memory added during inference (MB)", "{:.0f}")}
TITLES = {"map50": "mAP@0.5", "map": "mAP@0.5:0.95", "size": "File size", "time": "Model time per image", "f1": "F1 score", "mem": "Memory added during inference"}


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


def _draw(ax, metric):
    import matplotlib.patches as mp
    label, fmt = METRICS[metric]
    w = 0.22
    for gi in range(5):
        for fi, f in enumerate(COLORS):
            key = FP32[f] if gi == 0 else FP32_512[f] if gi == 1 else CURRENT[f][DESIGNS[gi - 2][1]]
            v = value(metric, key)
            x = gi + (fi - 1) * w
            ax.bar(x, v, w * 0.92, color=COLORS[f])
            ax.text(x, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(range(5)); ax.set_xticklabels(["FP32\n640 px", "FP32\n512 px"] + [d[0] for d in DESIGNS], fontsize=8)
    ax.set_ylabel(label); ax.yaxis.grid(True, color="#e6e6e6"); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.legend(handles=[mp.Patch(color=c, label=f) for f, c in COLORS.items()], frameon=False, ncol=3, fontsize=8, loc="upper right")


def era_chart_files(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    names = {}
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(10, 4.4))
        _draw(ax, m)
        fig.tight_layout(); fig.savefig(out_dir / f"era_{m}.png", dpi=150); plt.close(fig)
        names[m] = f"era_{m}.png"
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    for ax, m in zip(axs.flat, METRICS):
        _draw(ax, m)
    fig.tight_layout(); fig.savefig(out_dir / "era_all.png", dpi=130); plt.close(fig)
    names["all"] = "era_all.png"
    return names


def era_figure(metric, img_fn, names):
    return (f"<figure>{img_fn(names[metric], TITLES[metric] + ' by runtime and design')}<figcaption><b>{TITLES[metric]} by runtime and design.</b> "
            f"Blue: ONNX Runtime. Green: TFLite. Orange: OpenVINO. The first two groups are the unquantized model at 640 px and at 512 px; the other groups are the INT8 designs (per-channel), which take a 512 px input. "
            f"Compare the INT8 groups with FP32 at 512 px to see the effect of INT8 alone.</figcaption></figure>")


# ------------------------------------------------------------------ plain-language notes, numbers filled in from the data
def _v(metric, f, dk):
    return value(metric, CURRENT[f][dk])


def era_note(metric):
    ort, tfl, ov = "ONNX Runtime", "TFLite", "OpenVINO"
    if metric in ("map50", "map"):
        fmt = (lambda x: f"{x:.3f}") if metric == "map50" else (lambda x: f"{x:.4f}")
        nm = "mAP@0.5" if metric == "map50" else "mAP@0.5:0.95"
        lo = min(_v(metric, f, "m1") for f in COLORS)
        return (f"<p><b>How to read it.</b> The FP32 model scores {fmt(value(metric, 'm0_fp32'))} ({nm}). M2 (neck and head kept FP32) is the most accurate INT8 design in every runtime "
                f"({fmt(_v(metric, ort, 'm2'))}, {fmt(_v(metric, tfl, 'm2'))} and {fmt(_v(metric, ov, 'm2'))}), M3 is next, and the all-INT8 M1 is the least accurate (as low as {fmt(lo)}), "
                f"because every layer that stays FP32 cannot lose accuracy to rounding. The INT8 models also look at a smaller 512 px picture than the 640 px FP32 model: the unquantized network at 512 px scores {fmt(value(metric, 'f512_fp32'))}, so the smaller picture alone costs {(value(metric, 'm0_fp32') - value(metric, 'f512_fp32')) * 100:.1f} points of {nm} (times 100), "
                f"since there are fewer pixels to find small objects with. The three runtimes use different quantizers, so the same design scores slightly differently in each.</p>")
    if metric == "size":
        return (f"<p><b>What the sizes mean.</b> The size is the sum of the files a runtime needs: one file for ONNX Runtime, one .tflite file for TFLite, and the .xml graph plus the .bin weights for OpenVINO. "
                f"M1 is the smallest in every runtime ({_v('size', ort, 'm1'):.2f} MB ONNX Runtime, {_v('size', tfl, 'm1'):.2f} MB TFLite, {_v('size', ov, 'm1'):.2f} MB OpenVINO) because every weight is stored as 8 bits. "
                f"M3 is a little larger because its detection head stays FP32 ({_v('size', ort, 'm3'):.2f}, {_v('size', tfl, 'm3'):.2f} and {_v('size', ov, 'm3'):.2f} MB), and M2 is the largest INT8 design because the neck and the head both stay FP32 "
                f"({_v('size', ort, 'm2'):.2f}, {_v('size', tfl, 'm2'):.2f} and {_v('size', ov, 'm2'):.2f} MB). The FP32 model is {value('size', 'm0_fp32'):.2f} MB. "
                f"The input size does not change the file size, because the weights are the same at 512 px and 640 px. "
                f"The OpenVINO files are about 1 MB larger than the others, which matches the separate .xml graph file of about 1 MB.</p>")
    if metric == "time":
        t640 = {f: value("time", FP32[f]) for f in COLORS}
        t512 = {f: value("time", FP32_512[f]) for f in COLORS}
        best = {f: min(_v("time", f, d) for d in ("m1", "m2", "m3")) for f in COLORS}
        return (f"<p><b>How to read the speed.</b> Two things make the INT8 models faster than the 640 px FP32 model: the smaller 512 px picture and INT8 itself. The two FP32 groups separate them. "
                f"Going from 640 px to 512 px alone takes ONNX Runtime from {t640[ort]:.1f} to {t512[ort]:.1f} ms, TFLite from {t640[tfl]:.1f} to {t512[tfl]:.1f} ms and OpenVINO from {t640[ov]:.1f} to {t512[ov]:.1f} ms. "
                f"INT8 on top of that brings the best design to {best[ort]:.1f}, {best[tfl]:.1f} and {best[ov]:.1f} ms, which is {t512[ort] / best[ort]:.2f}x, {t512[tfl] / best[tfl]:.2f}x and {t512[ov] / best[ov]:.2f}x faster than FP32 at the same 512 px. "
                f"<b>The order of the designs is not the same in every runtime.</b> In ONNX Runtime, M2 ({_v('time', ort, 'm2'):.1f} ms) is faster than the all-INT8 M1 ({_v('time', ort, 'm1'):.1f} ms): "
                f"the neck's resize and concatenate steps need extra conversion work when they are INT8, which costs more than the INT8 convolutions save (a per-operation profile of the network showed this). "
                f"TFLite and OpenVINO behave the other way round: M1 is fastest ({_v('time', tfl, 'm1'):.1f} ms and {_v('time', ov, 'm1'):.1f} ms) and M2 slowest ({_v('time', tfl, 'm2'):.1f} and {_v('time', ov, 'm2'):.1f} ms), "
                f"consistent with the FP32 neck of M2 being the expensive part there.</p>")
    if metric == "mem":
        m640 = {f: value("mem", FP32[f]) for f in COLORS}
        m512 = {f: value("mem", FP32_512[f]) for f in COLORS}
        tp = E.J("tflite_memory_probe.json")
        return (f"<p><b>How to read the memory.</b> This is the extra memory the process needs for the model while it runs, above a process that has only loaded the runtime: weights, intermediate tensors (activations) and the runtime's working buffers. "
                f"Memory grows with the number of pixels, so compare the INT8 models with FP32 at the same 512 px, not with the 640 px model: FP32 at 640 px needs {m640[ort]:.0f} MB in ONNX Runtime, {m640[tfl]:.0f} MB in TFLite and {m640[ov]:.0f} MB in OpenVINO, "
                f"but at 512 px only {m512[ort]:.0f}, {m512[tfl]:.0f} and {m512[ov]:.0f} MB. Against that 512 px reference, the all-INT8 M1 needs {_v('mem', ort, 'm1'):.0f} MB in ONNX Runtime ({(_v('mem', ort, 'm1') / m512[ort] - 1) * 100:+.0f}%) "
                f"and {_v('mem', tfl, 'm1'):.0f} MB in TFLite ({(_v('mem', tfl, 'm1') / m512[tfl] - 1) * 100:+.0f}%): INT8 tensors are smaller, but the saving is modest. M2 and M3 need more than M1 because their FP32 parts keep 32-bit tensors. "
                f"<b>TFLite's large figures come from its default XNNPACK engine.</b> When a model is loaded, XNNPACK sets aside its own working buffers ({tp['m0_fp32']['default']['after_load_mb']:.0f} MB at load for the 640 px FP32 model). "
                f"With XNNPACK switched off, the same model peaks at {tp['m0_fp32']['no_xnnpack']['peak_mb']:.0f} MB instead of {tp['m0_fp32']['default']['peak_mb']:.0f} MB, but it is much slower, so the benchmark uses the default engine. "
                f"ONNX Runtime reserves its buffers while running instead, so the &ldquo;memory at load&rdquo; columns are not comparable between runtimes. "
                f"<b>OpenVINO is different:</b> its INT8 models use more memory than its FP32 model at the same size ({_v('mem', ov, 'm1'):.0f} MB for M1 against {m512[ov]:.0f} MB). "
                f"A likely cause is the memory OpenVINO uses to read and compile a quantized graph, which contains many extra conversion nodes, but this was not verified. "
                f"Peak process memory includes that compile step, so it is the right number for a device that must load the model, and the wrong one for judging the steady-state size of the activations.</p>")
    if metric == "f1":
        return (f"<p><b>How to read F1.</b> F1 combines precision (how many reported objects are right) and recall (how many real objects are found) at confidence threshold 0.25. FP32 scores {value('f1', 'm0_fp32'):.3f}; "
                f"the quantized models lose recall and keep most of their precision, so F1 falls by {(value('f1', 'm0_fp32') - _v('f1', ort, 'm2')) * 100:.1f} points for ONNX Runtime M2 and {(value('f1', 'm0_fp32') - _v('f1', ort, 'm1')) * 100:.1f} for M1. "
                f"The designs rank as in mAP.</p>")
    return ""


def _row(f, dk, label):
    key = CURRENT[f][dk]
    k = ACCKEY.get(key, key)
    a = acc[k]
    t640, t512 = LAT[FP32[f]]["latency_ms_median"], LAT[FP32_512[f]]["latency_ms_median"]
    m512 = mem[FP32_512[f]]["peak_mb"]
    return [f, label, f"{a['mAP50-95']:.4f}", f"{a['mAP50-95'] / BASE * 100:.1f}%", f"{a['size_mb']:.2f}", f"{LAT[key]['latency_ms_median']:.1f}",
            f"{t640 / LAT[key]['latency_ms_median']:.2f}x", f"{t512 / LAT[key]['latency_ms_median']:.2f}x", f"{prf[k]['F1@0.25']:.3f}", f"{mem[key]['peak_mb']:.0f}",
            f"{(mem[key]['peak_mb'] / m512 - 1) * 100:+.0f}%"]


def eras_html(img_fn, names):
    rows = []
    for f in COLORS:
        base = FP32[f]
        k = ACCKEY.get(base, base)
        k5 = ACCKEY.get(FP32_512[f], FP32_512[f])
        rows.append([f, "FP32 (640 px)", f"{acc[k]['mAP50-95']:.4f}", "100.0%", f"{acc[k]['size_mb']:.2f}", f"{LAT[base]['latency_ms_median']:.1f}", "1.00x", "-",
                     f"{prf[k]['F1@0.25']:.3f}", f"{mem[base]['peak_mb']:.0f}", "-"])
        rows.append([f, "FP32 (512 px)", f"{acc[k5]['mAP50-95']:.4f}", f"{acc[k5]['mAP50-95'] / BASE * 100:.1f}%", f"{acc[k5]['size_mb']:.2f}", f"{LAT[FP32_512[f]]['latency_ms_median']:.1f}",
                     f"{LAT[base]['latency_ms_median'] / LAT[FP32_512[f]]['latency_ms_median']:.2f}x", "1.00x", f"{prf[k5]['F1@0.25']:.3f}", f"{mem[FP32_512[f]]['peak_mb']:.0f}", "0%"])
        rows += [_row(f, dk, d_label.replace("\n", " ") + " (512 px)") for d_label, dk in DESIGNS]
    tab = E.table(["Runtime", "Model", "mAP50-95", "Retained", "Size (MB)", "Model ms", "Speed-up vs FP32 640 px", "Speed-up vs FP32 512 px", "F1", "Peak memory added (MB)", "Memory vs FP32 512 px"], rows,
                  "The models in all three runtimes (per-channel INT8). Retained = share of the FP32 (640 px) mAP50-95. The two FP32 rows are the unquantized network at 640 px and at 512 px; comparing the INT8 models with the 512 px row shows the effect of INT8 alone.")
    pp = E.PP
    return f"""<h2>The models</h2>
<p>The benchmark compares the original FP32 YOLOv5n (nano) with three INT8 designs in three runtimes. The Detect head (the last layer) is untouched in every model.</p>
<ul>
<li><b>M1</b> everything INT8; <b>M2</b> backbone INT8 with neck and head FP32; <b>M3</b> backbone and neck INT8 with the head FP32. Each is built per-channel and per-tensor (the main charts show per-channel; the tables below list both).</li>
<li>Runtimes: <b>ONNX Runtime</b> (ONNX file, percentile calibration), <b>TFLite</b> (one .tflite file per model, min-max calibration, run with LiteRT) and <b>OpenVINO</b> (NNCF quantization).</li>
<li>The INT8 models take a 512 x 512 picture; the original FP32 model takes 640 x 640 and is also measured at 512 x 512, so the effect of INT8 can be separated from the effect of the smaller picture. Calibration uses 500 train2017 pictures; testing uses all 5000 val2017 pictures with the official COCO evaluation.</li>
<li>Box coordinates are normalised to 0-1 so that the all-INT8 model works. Post-processing time is reported separately (a compiled routine takes {pp['post_ms_compiled_conf0.25']:.1f} ms against {pp['post_ms_numpy_conf0.25']:.1f} ms for the reference one at confidence 0.25).</li>
</ul>

<figure>{img_fn(names['all'], 'All benchmarks by runtime and design')}<figcaption><b>All six benchmarks by runtime and design.</b> Blue: ONNX Runtime. Green: TFLite. Orange: OpenVINO. The FP32 group is the 640 px original in each runtime; the others are the INT8 designs at 512 px.</figcaption></figure>
{tab}"""
