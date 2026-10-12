"""HTML sections shared by report.html and benchmark_summary.html: measurement protocol, matched-calibration ONNX/TFLite comparison,
three-runtime result tables, why INT8 is not 2x/4x, ReLU-activation models (R55, R80). Reads results/*.json. Import SECTION_* and make_chart()."""
import html
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
J = lambda n: json.loads((R / n).read_text())
env, lat, mem = J("environment.json"), J("latency.json"), J("memory.json")
acc = {}
for f in ("accuracy.json", "accuracy_tflite.json", "accuracy_ov.json"):
    acc.update(J(f))
cm = J("calib_matched.json")
BASE = acc["m0_fp32"]["mAP50-95"]
proto = env["benchmark_protocol"]
L = lambda k: lat[k]["latency_ms_median"]
BL = L("m0_fp32")
TFL_BL = L("tflite_m0_fp32")


def table(cols, rows, caption):
    th = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><caption>{caption}</caption><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def row(key, label, fmt, acc_key=None, ref=None):
    a = acc.get(acc_key or key)
    if key not in lat or a is None:
        return None
    ref = ref or BL
    m = mem.get(key, {})
    mp = m.get("peak_mb")
    return [fmt, label, f"{a['mAP50-95']:.4f}", f"{a['mAP50-95'] / BASE * 100:.1f}%", f"{a['size_mb']:.2f}",
            f"{L(key):.1f}", f"{lat[key]['latency_ms_p95']:.1f}", f"{ref / L(key):.2f}x", f"{lat[key]['round_spread_pct']:.0f}%",
            "" if mp is None else f"{mp:.0f}"]


COLS = ["Runtime", "Model", "mAP50-95", "Retained", "Size (MB)", "Median ms", "p95 ms", "Speed-up", "Round spread", "Peak mem added (MB)"]
groups = {
    "ORT": [("m0_fp32", "FP32 baseline"), ("m1_all_int8_pc_norm", "M1 all INT8, per-channel"), ("m1_all_int8_pt_norm", "M1 all INT8, per-tensor"),
            ("m2_backbone_int8_pc", "M2 per-channel"), ("m2_backbone_int8_pt", "M2 per-tensor"),
            ("m3_backbone_neck_int8_pc", "M3 per-channel"), ("m3_backbone_neck_int8_pt", "M3 per-tensor"),
            ("h_hybrid_s55_fp32", "R55 v1 FP32"), ("h_hybrid_s55_v2_fp32", "R55 v2 FP32")],
    "OpenVINO": [("ov_m0_fp32", "FP32 baseline"), ("ovq_m1_mixed", "M1 SiLU, NNCF"), ("ovq_m2_mixed", "M2 SiLU, NNCF"), ("ovq_m3_mixed", "M3 SiLU, NNCF"),
                 ("ov_h_hybrid_s55_v2_fp32", "R55 v2 FP32"),
                 ("ovq_hybrid_s55_m1_mixed", "R55 v1 M1, NNCF"), ("ovq_hybrid_s55_m2_mixed", "R55 v1 M2, NNCF"), ("ovq_hybrid_s55_m3_mixed", "R55 v1 M3, NNCF"),
                 ("ovq_hybrid_s55_v2_m1_mixed", "R55 v2 M1, NNCF"), ("ovq_hybrid_s55_v2_m2_mixed", "R55 v2 M2, NNCF"), ("ovq_hybrid_s55_v2_m3_mixed", "R55 v2 M3, NNCF"),
                 ("ovq_hybrid_s80_m1_mixed", "R80 M1, NNCF"), ("ovq_hybrid_s80_m2_mixed", "R80 M2, NNCF"), ("ovq_hybrid_s80_m3_mixed", "R80 M3, NNCF"),
                 ("ovq_f512_m1_mixed", "512 px M1, NNCF"), ("ovq_f512_m3_mixed", "512 px M3, NNCF"), ("ovq_f448_m1_mixed", "448 px M1, NNCF"), ("ovq_f448_m3_mixed", "448 px M3, NNCF"),
                 ("ovq_f416_m1_mixed", "416 px M1, NNCF"), ("ovq_f416_m3_mixed", "416 px M3, NNCF"),
                 ("ovq_r55_512_m1_mixed", "R55 at 512 px M1 (fine-tuned at 512)"), ("ovq_r55_512_m3_mixed", "R55 at 512 px M3 (fine-tuned at 512)"),
                 ("ovq_s80_512_m1_mixed", "R80 at 512 px M1"), ("ovq_s80_512_m3_mixed", "R80 at 512 px M3"),
                 ("ovq_ft_448_m1_mixed", "448 px fine-tuned M1"), ("ovq_ft_448_m3_mixed", "448 px fine-tuned M3")],
    "TFLite": [("tflite_m0_fp32", "FP32 baseline"), ("tflite_m1_all_int8_pc", "M1 per-channel"), ("tflite_m1_all_int8_pt", "M1 per-tensor"),
               ("tflite_m2_backbone_int8_pc", "M2 per-channel"), ("tflite_m2_backbone_int8_pt", "M2 per-tensor"),
               ("tflite_m3_backbone_neck_int8_pc", "M3 per-channel"), ("tflite_m3_backbone_neck_int8_pt", "M3 per-tensor")],
}
ACC_KEY = {"ov_m0_fp32": "m0_fp32", "ov_h_hybrid_s55_v2_fp32": "hybrid_s55_v2_fp32", "h_hybrid_s55_fp32": "hybrid_s55_fp32", "h_hybrid_s55_v2_fp32": "hybrid_s55_v2_fp32"}
tables = ""
for i, (rt, items) in enumerate(groups.items(), 1):
    ref = TFL_BL if rt == "TFLite" else (L("ov_m0_fp32") if rt == "OpenVINO" and "ov_m0_fp32" in lat else BL)
    rows = [r for r in (row(k, lab, rt, ACC_KEY.get(k), ref) for k, lab in items) if r]
    cap = {"ORT": "ONNX Runtime", "OpenVINO": "OpenVINO (CPU plugin)", "TFLite": "TFLite (XNNPACK)"}[rt]
    tables += table(COLS, rows, f"Table {i + 1}. {cap}. Speed-up is against the FP32 baseline of the same runtime.")

# ---- ONNX versus TFLite, calibration matched
mm = []
for k, lab in (("m1_all_int8_pc", "M1 per-channel"), ("m1_all_int8_pt", "M1 per-tensor"), ("m2_backbone_int8_pc", "M2 per-channel"),
               ("m2_backbone_int8_pt", "M2 per-tensor"), ("m3_backbone_neck_int8_pc", "M3 per-channel"), ("m3_backbone_neck_int8_pt", "M3 per-tensor")):
    o = acc[k + "_norm"]["mAP50-95"] if k.startswith("m1") else acc[k]["mAP50-95"]
    t = acc["tflite_" + k]["mAP50-95"]
    c = cm[k]["mAP50-95"]
    mm.append([lab, f"{o:.4f}", f"{c:.4f}", f"{t:.4f}", f"{(o - t) * 100:+.2f}", f"{(c - t) * 100:+.2f}"])
T_MM = table(["Design", "ONNX, percentile", "ONNX, min-max", "TFLite (min-max)", "ONNX pct. minus TFLite (pts)", "ONNX min-max minus TFLite (pts)"], mm,
             "Table 5. The same designs with the calibration method matched. Only the calibration method changes between columns 2 and 3.")

LBL = lambda k: {kk: lab for g_ in groups.values() for kk, lab in g_}.get(k, k)
best_ov = max((k for k in lat if k.startswith("ovq_") and k in acc), key=lambda k: L("ov_m0_fp32") / L(k) * (acc[k]["mAP50-95"] / BASE >= 0.95))
cpu = env["cpu"].replace("(R)", "").replace("(TM)", "")



def make_chart(out_dir):
    """Speed-up versus retained accuracy for every model, three runtimes. Returns the file name."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5))
    sets = [("ONNX Runtime", "#1f4e9e", [k for k, _ in groups["ORT"]], BL), ("OpenVINO", "#c0551b", [k for k, _ in groups["OpenVINO"]], L("ov_m0_fp32")),
            ("TFLite", "#1b7f5c", [k for k, _ in groups["TFLite"]], TFL_BL)]
    for name, col, keys, ref in sets:
        pts = [(ref / L(k), acc[ACC_KEY.get(k, k)]["mAP50-95"] / BASE * 100, k) for k in keys if k in lat and ACC_KEY.get(k, k) in acc]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=col, label=name)
        for x, y, k in pts:
            if "hybrid" in k and "m3" in k or k.endswith(("m1_mixed",)) and "hybrid" not in k:
                ax.annotate(k.replace("ovq_", "").replace("_mixed", "").replace("hybrid_s55_v2", "R55").replace("hybrid_s55", "R55 v1").replace("hybrid_s80", "R80").replace("m1", "M1").replace("m3", "M3"), (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(2, color="#999", ls="--", lw=1); ax.text(2.01, 84.5, "2x", color="#666", fontsize=9)
    ax.axhline(95, color="#999", ls=":", lw=1); ax.text(0.99, 95.3, "95% retained", color="#666", fontsize=9)
    ax.set_xlabel("speed-up over the FP32 model of the same runtime"); ax.set_ylabel("accuracy retained (mAP50-95, % of FP32)")
    ax.yaxis.grid(True, color="#e6e6e6"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(Path(out_dir) / "8_speed_accuracy_tradeoff.png", dpi=150); plt.close(fig)
    return "8_speed_accuracy_tradeoff.png"


PP = J("postprocess.json")
def _e2e(label, model_ms, post_key, size):
    pre = PP["pre_ms"] * (size / 640) ** 2
    post = PP[post_key]
    return label, pre, model_ms, post, pre + model_ms + post
_rows = [_e2e("FP32 baseline, 640 px, numpy post-processing, threshold 0.001", L("ov_m0_fp32"), "post_ms_numpy_conf0.001", 640),
         _e2e("FP32 baseline, 640 px, numpy post-processing, threshold 0.25", L("ov_m0_fp32"), "post_ms_numpy_conf0.25", 640),
         _e2e("INT8 M3 (OpenVINO), 640 px, compiled post-processing, threshold 0.25", L("ovq_m3_mixed"), "post_ms_compiled_conf0.25", 640),
         _e2e("INT8 M3 (OpenVINO), 512 px, compiled post-processing, threshold 0.25", L("ovq_f512_m3_mixed"), "post_ms_compiled_conf0.25", 512),
         _e2e("INT8 M3 (OpenVINO), 448 px, compiled post-processing, threshold 0.25", L("ovq_f448_m3_mixed"), "post_ms_compiled_conf0.25", 448)]
_ref = _rows[1][4]
E2E_TABLE = table(["Pipeline", "Pre (ms)", "Model (ms)", "Post (ms)", "Total (ms)", "vs FP32 baseline at threshold 0.25"],
                  [[r[0], f"{r[1]:.1f}", f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.1f}", f"{_ref / r[4]:.2f}x"] for r in _rows],
                  "End-to-end time per image on the benchmark CPU (4 pinned cores). Preprocessing at other sizes is scaled by pixel count; post-processing time is that of the same detections as measured at 640 px (a conservative figure for smaller inputs).")

SECTION = f"""<h2>1. Measurement protocol</h2>
<div class="tablewrap"><table class="text"><caption>Table 1. How speed and memory were measured.</caption><tbody>
<tr><td>CPU</td><td>{html.escape(cpu)}: {env['physical_cores']} physical cores, {env['logical_processors']} logical processors; AVX-512 with VNNI (fast INT8 dot products).</td></tr>
<tr><td>Cores used</td><td>{proto['threads']} threads. {html.escape(proto['thread_placement'])}. Priority: {proto['process_priority']}.</td></tr>
<tr><td>Batch</td><td>{proto['batch_size']} image per call. Input {html.escape(proto['input'])}. Pre- and post-processing are timed separately and are not in the headline figure.</td></tr>
<tr><td>Warm-up</td><td>{proto['warmup_calls_per_round']} untimed calls before each timed measurement. The first call is slow (memory allocation, kernel selection); the time settles after 1 to 2 calls, so 20 is a wide margin.</td></tr>
<tr><td>Timed runs</td><td>{proto['timed_calls_per_round']} calls per measurement, cycling over 10 fixed images.</td></tr>
<tr><td>Rounds</td><td>{proto['rounds']} rounds. {html.escape(proto['round_order'])}. Headline = {html.escape(proto['statistic'])}. The round spread column is (max minus min) over the median.</td></tr>
<tr><td>Machine state</td><td>Idle apart from the benchmark, on AC power, Windows power plan: Balanced. Latency was never measured while training ran (that gave an FP32 baseline of 63 ms instead of about 25 ms and was discarded).</td></tr>
<tr><td>Memory</td><td>Increase in resident memory over a process that already has the runtime loaded, sampled every 2 ms during inference, median of 3 runs.</td></tr>
<tr><td>Software</td><td>ONNX Runtime {env['packages_main_env']['onnxruntime']}, OpenVINO {html.escape(env['packages_main_env']['openvino'].split('-')[0])}, TensorFlow {env['packages_tensorflow_env']['tensorflow']}, Python {env['python']}.</td></tr>
</tbody></table></div>

<h2>2. Why ONNX and TFLite differ for the same design</h2>
<p>The FP32 baselines are identical (mAP50-95 {BASE:.4f}). Quantization is not: ONNX Runtime calibrates with percentile clipping, while the TFLite converter only offers min-max. Percentile ignores rare outliers and so gives finer steps for the typical values. Rebuilding the ONNX models with min-max calibration removes most of the gap (Table 5): the remaining differences are in the range of {min(float(r[5]) for r in mm):+.2f} to {max(float(r[5]) for r in mm):+.2f} mAP points, while the percentile versions are {min(float(r[4]) for r in mm):+.2f} to {max(float(r[4]) for r in mm):+.2f} points above TFLite. The gap is a calibration effect, not a property of the format.</p>
{T_MM}

<h2>3. Speed and accuracy, every model</h2>
<p>All three runtimes, same protocol. The FP32 baseline takes {BL:.1f} ms per image with ONNX Runtime (batch 1, 4 pinned cores), {L('ov_m0_fp32') if 'ov_m0_fp32' in lat else float('nan'):.1f} ms with OpenVINO and {TFL_BL:.1f} ms with TFLite.</p>
{tables}

<h2>4. Why INT8 is not 2x or 4x faster here</h2>
<ul>
<li><b>The convolutions are only part of the time.</b> In FP32 the SiLU activation is fused into the convolution. In INT8 it is a separate sigmoid and multiply on every layer, and its cost grows with the size of the activation tensor, which is largest in the first layers.</li>
<li><b>Other operators do not speed up:</b> in ONNX Runtime the Resize and Concat steps of the neck need extra requantization work, so quantizing the neck (M3) is slower than leaving it in FP32 (M2). This was seen in the per-operation profile (<code>results/profile.json</code>).</li>
<li><b>Result with SiLU:</b> INT8 reaches {BL / L('m2_backbone_int8_pc'):.2f}x at best in ONNX Runtime and {L('ov_m0_fp32') / L('ovq_m1_mixed'):.2f}x in OpenVINO (Tables 2 and 3), and no more in TFLite on this CPU.</li>
<li><b>Result with ReLU in the largest layers (Detect head bit-identical):</b> the INT8 convolutions fuse their activation. With 12 of 57 activations as ReLU (R55) OpenVINO reaches {L('ov_m0_fp32') / L('ovq_hybrid_s55_v2_m3_mixed'):.2f}x (head FP32) and {L('ov_m0_fp32') / L('ovq_hybrid_s55_v2_m1_mixed'):.2f}x (all INT8) over its own FP32 baseline at 92 to 93% of the accuracy. With 29 of 57 (R80) it reaches {L('ov_m0_fp32') / L('ovq_hybrid_s80_m3_mixed'):.2f}x and {L('ov_m0_fp32') / L('ovq_hybrid_s80_m1_mixed'):.2f}x, at about 88 to 89% of the accuracy. The 2x mark is reached by the all-INT8 R80 model only at the cost of about 12% of the accuracy. A smaller input size does better: 2x at 95% or more of the accuracy (Section 6).</li>
<li>A 4x gain needs hardware where INT8 is not just faster but the only path (an NPU); this CPU cannot show that.</li>
</ul>

<h2>5. The R55 and R80 models (ReLU activations)</h2>
<p>R55 changed 12 of the 57 activations (layers 0 to 4 and 17, about 55% of the activation work) from SiLU to ReLU; R80 changed 29 (81% of the activation work, layers 0 to 6, 13, 17 and 20) and the network was fine-tuned with distillation from the original model through a frozen Detect head. v1 used an abrupt swap and short training; v2 used a gradual swap, ground-truth loss, EMA and 24 epochs. The Detect head weights are identical to the original (verified by comparing every parameter).</p>
<ul>
<li>FP32 accuracy: R55 v1 {acc['hybrid_s55_fp32']['mAP50-95']:.4f} ({acc['hybrid_s55_fp32']['mAP50-95'] / BASE * 100:.1f}% retained), R55 v2 {acc['hybrid_s55_v2_fp32']['mAP50-95']:.4f} ({acc['hybrid_s55_v2_fp32']['mAP50-95'] / BASE * 100:.1f}%), R80 {acc['hybrid_s80_fp32']['mAP50-95']:.4f} ({acc['hybrid_s80_fp32']['mAP50-95'] / BASE * 100:.1f}%). Replacing more activations buys speed and costs accuracy: each extra step is roughly 0.3x speed for 3 to 4 points.</li>
<li>Fastest model keeping 95% or more of the accuracy: <b>{LBL(best_ov)}</b>, {L('ov_m0_fp32') / L(best_ov):.2f}x versus the OpenVINO FP32 baseline, at {acc[best_ov]['mAP50-95'] / BASE * 100:.1f}% retained.</li>
<li>For comparison, the best SiLU model that keeps 97% or more (OpenVINO M3, NNCF) is {L('ov_m0_fp32') / L('ovq_m3_mixed'):.2f}x over OpenVINO FP32. The trade is roughly 6 points of retained accuracy for about 0.4x more speed.</li>
</ul>

<h2>6. A smaller input size</h2>
<p>Computation falls with the square of the input size, so the plain network (SiLU, no retraining, Detect head untouched) was exported at 512, 448 and 416 pixels, quantized with the same OpenVINO procedure (calibration images letterboxed to the same size) and scored on all 5000 images (Table 3). Accuracy stays high because most COCO objects are large enough, and speed rises more than with any activation change:</p>
<ul>
<li>640 px (reference): M3 {L('ov_m0_fp32') / L('ovq_m3_mixed'):.2f}x at {acc['ovq_m3_mixed']['mAP50-95'] / BASE * 100:.1f}% retained.</li>
<li>512 px: M3 {L('ov_m0_fp32') / L('ovq_f512_m3_mixed'):.2f}x at {acc['ovq_f512_m3_mixed']['mAP50-95'] / BASE * 100:.1f}% (all-INT8 M1 {L('ov_m0_fp32') / L('ovq_f512_m1_mixed'):.2f}x at {acc['ovq_f512_m1_mixed']['mAP50-95'] / BASE * 100:.1f}%): <b>2x at 95% or more of the accuracy.</b></li>
<li>448 px: M3 {L('ov_m0_fp32') / L('ovq_f448_m3_mixed'):.2f}x at {acc['ovq_f448_m3_mixed']['mAP50-95'] / BASE * 100:.1f}%.</li>
<li>416 px: M3 {L('ov_m0_fp32') / L('ovq_f416_m3_mixed'):.2f}x at {acc['ovq_f416_m3_mixed']['mAP50-95'] / BASE * 100:.1f}%.</li>
</ul>
<p>Speed-ups are against the 640 px FP32 model, since that is the baseline the lower-resolution models replace. Small objects are the ones lost first, so this is the lever to choose only if the application's objects are not tiny.</p>
<p><b>Combinations that did not help.</b> The ReLU models were also fine-tuned and exported at 512 px (R55 at 512: {L('ov_m0_fp32') / L('ovq_r55_512_m3_mixed'):.2f}x at {acc['ovq_r55_512_m3_mixed']['mAP50-95'] / BASE * 100:.1f}%), and the plain network was fine-tuned at 448 px with the head frozen ({acc['ovq_ft_448_m3_mixed']['mAP50-95'] / BASE * 100:.1f}% retained after INT8, against {acc['ovq_f448_m3_mixed']['mAP50-95'] / BASE * 100:.1f}% without fine-tuning). Both are worse than simply lowering the resolution of the original network: with the head frozen, retraining on 9,500 images moved the backbone away from what the head expects more than it helped.</p>

<h2>7. End-to-end time per image</h2>
{E2E_TABLE}
<p>The model is only part of the time per image. The reference post-processing (a numpy implementation of non-maximum suppression) took {PP['post_ms_numpy_conf0.001']:.1f} ms at the evaluation confidence threshold of 0.001 and {PP['post_ms_numpy_conf0.25']:.1f} ms at the usual deployment threshold of 0.25. Replacing it with a compiled non-maximum-suppression routine gives identical detections on {PP['identical_images_of_100']} of 100 test images and takes {PP['post_ms_compiled_conf0.001']:.1f} ms and {PP['post_ms_compiled_conf0.25']:.1f} ms. Preprocessing (letterbox and conversion) takes {PP['pre_ms']:.1f} ms.</p>

<h2>8. Limitations</h2>
<ul>
<li>All timings are from a laptop CPU; an NPU's ranking and speed-ups can differ. The exact target board is not known.</li>
<li>Round-to-round spread on this laptop is large (see the Round spread column), so differences of a few percent between INT8 variants are not meaningful; differences to FP32 are.</li>
<li>The R55 and R80 models are not available as TFLite; TFLite results cover the SiLU designs only.</li>
<li>OpenVINO INT8 models were quantized with NNCF (mixed preset, 500 calibration images); results for ONNX Runtime QDQ and OpenVINO are different quantizers and runtimes, not one model on two engines.</li>
</ul>
"""


def embed(img_b64, numbered_from=None):
    """SECTION with the standalone numbering removed and the trade-off chart inserted (img_b64 = <img> tag)."""
    out = re.sub(r"<h2>\d+\. ", "<h2>", SECTION)
    out = re.sub(r"Table \d+\. ", "", out)
    h = "<h2>Speed and accuracy, every model</h2>"
    fig = ("<figure>" + img_b64 + "<figcaption><b>Speed against accuracy.</b> Each dot is one model; the further right and up, the better. "
           "The dashed line marks a 2x speed-up and the dotted line 95% of the FP32 accuracy; models in the upper right of both lines are fast and accurate.</figcaption></figure>")
    return out.replace(h, h + chr(10) + fig)


DARK_CSS = """
#themeBtn { position:fixed; top:14px; right:14px; z-index:50; font:600 13px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; padding:7px 12px; border:1px solid #bbb; border-radius:18px; background:#fff; color:#1a1a1a; cursor:pointer; }
@media print { #themeBtn { display:none; } }
html[data-theme="dark"], html[data-theme="dark"] body { background:#16181c !important; color:#e4e6ea !important; color-scheme:dark; }
html[data-theme="dark"] #themeBtn { background:#23262c; color:#e4e6ea; border-color:#555; }
html[data-theme="dark"] h1, html[data-theme="dark"] h2, html[data-theme="dark"] h3, html[data-theme="dark"] caption, html[data-theme="dark"] b, html[data-theme="dark"] dt, html[data-theme="dark"] figcaption b { color:#f4f5f7 !important; }
html[data-theme="dark"] .meta, html[data-theme="dark"] footer, html[data-theme="dark"] figcaption, html[data-theme="dark"] .def, html[data-theme="dark"] dd { color:#a6abb3 !important; }
html[data-theme="dark"] .meta, html[data-theme="dark"] footer, html[data-theme="dark"] nav, html[data-theme="dark"] .summary { border-color:#3a3e46 !important; }
html[data-theme="dark"] thead th { border-color:#d0d3d9 !important; }
html[data-theme="dark"] tbody tr { border-color:#2d3036 !important; } html[data-theme="dark"] tbody tr:last-child { border-bottom-color:#d0d3d9 !important; }
html[data-theme="dark"] code { background:#262a31 !important; color:#e4e6ea !important; }
html[data-theme="dark"] a { color:#8ab4f8; } html[data-theme="dark"] nav { background:#16181c !important; }
html[data-theme="dark"] figure img { background:#fff; border-radius:6px; }
html[data-theme="dark"] .read { background:#1d2025 !important; border-color:#3a3e46 !important; }
"""
DARK_JS = """<button id="themeBtn" type="button" aria-label="Switch between light and dark mode">Dark mode</button>
<script>
(function(){var h=document.documentElement,b=document.getElementById('themeBtn'),k='reportTheme';
function set(t){h.setAttribute('data-theme',t);b.textContent=t==='dark'?'Light mode':'Dark mode';try{localStorage.setItem(k,t);}catch(e){}}
var t='light';try{t=localStorage.getItem(k)||'light';}catch(e){}
set(t);b.addEventListener('click',function(){set(h.getAttribute('data-theme')==='dark'?'light':'dark');});})();
</script>"""


def add_dark_mode(page):
    """Insert the dark-mode style block and toggle button into a finished page."""
    page = page.replace("</style></head><body>", DARK_CSS + "</style></head><body>" + DARK_JS, 1)
    return page
