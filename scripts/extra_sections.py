"""HTML sections shared by report.html and benchmark_summary.html: measurement protocol, matched-calibration ONNX/TFLite comparison,
three-runtime result tables, why INT8 is not 2x/4x, TFLite recipe and runtime. Reads results/*.json. Import SECTION_* and make_chart()."""
import html
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
J = lambda n: json.loads((R / n).read_text())
RP = J("tflite_runtime_probe.json")                # TFLite models timed with two runtimes (model time only)
env, lat, mem = J("environment.json"), J("latency.json"), J("memory.json")
acc = {}
for f in ("accuracy.json", "accuracy_tflite.json", "accuracy_ov.json"):
    acc.update(J(f))
cm = J("calib_matched.json")
ST = J("study_inputs.json")                       # mAP50-95 of the 640 px models behind the calibration study (the model files are archived)
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
    size = "512" if "512" in key else "640"                    # network input size of this model
    ms, p95, spread = L(key), lat[key]["latency_ms_p95"], f"{lat[key]['round_spread_pct']:.0f}%"
    if key.endswith("_io8"):            # integer input: the benchmark runner's Python quantization of the picture is not model time, so the model-only probe is shown
        pr = RP[key.replace("tflite_", "")]["litert"]
        ms, p95, spread, label = pr["median_ms"], pr["p95_ms"], "n/a", label + " (model time only)"
    return [fmt, label, size, f"{a['mAP50-95']:.4f}", f"{a['mAP50-95'] / BASE * 100:.1f}%", f"{a['size_mb']:.2f}",
            f"{ms:.1f}", f"{p95:.1f}", f"{ref / ms:.2f}x", spread,
            "n/a" if mp is None else f"{mp:.0f}"]


COLS = ["Runtime", "Model", "Input (px)", "mAP50-95", "Retained", "Size (MB)", "Median ms", "p95 ms", "Speed-up", "Round spread", "Peak mem added (MB)"]
groups = {
    "ORT": [("m0_fp32", "FP32 baseline")] + [(f"s512_{d}_{g}", f"{n} {gl}") for d, n in (("m1_all_int8", "M1 all INT8,"), ("m2_backbone_int8", "M2"), ("m3_backbone_neck_int8", "M3")) for g, gl in (("pc", "per-channel"), ("pt", "per-tensor"))],
    "OpenVINO": [("ov_m0_fp32", "FP32 baseline")] + [(f"ovq_f512_m{i}_mixed", f"M{i}, NNCF") for i in (1, 2, 3)],
    "TFLite": [("tflite_m0_fp32", "FP32 baseline"), ("tflite_s512_m0_fp32", "FP32, 512 px input")]
              + [(f"tflite_t512_{d}_{g}", f"{n} {gl}") for d, n in (("m1_all_int8", "M1 all INT8,"), ("m2_backbone_int8", "M2"), ("m3_backbone_neck_int8", "M3")) for g, gl in (("pc", "per-channel"), ("pt", "per-tensor"))]
              + [("tflite_t512_m1_all_int8_pc_io8", "M1 per-channel, integer input/output"), ("tflite_t512_m1_all_int8_pt_io8", "M1 per-tensor, integer input/output")],
}
ACC_KEY = {"ov_m0_fp32": "m0_fp32"}
tables = ""
for i, (rt, items) in enumerate(groups.items(), 1):
    ref = TFL_BL if rt == "TFLite" else (L("ov_m0_fp32") if rt == "OpenVINO" and "ov_m0_fp32" in lat else BL)
    rows = [r for r in (row(k, lab, rt, ACC_KEY.get(k), ref) for k, lab in items) if r]
    cap = {"ORT": "ONNX Runtime", "OpenVINO": "OpenVINO (CPU plugin)", "TFLite": "TFLite (XNNPACK)"}[rt]
    tables += table(COLS, rows, f"Table {i + 1}. {cap}. Speed-up is against the 640 px FP32 baseline of the same runtime; the Input column gives each model's input size.")

# ---- ONNX versus TFLite, calibration matched
mm = []
for k, lab in (("m1_all_int8_pc", "M1 per-channel"), ("m1_all_int8_pt", "M1 per-tensor"), ("m2_backbone_int8_pc", "M2 per-channel"),
               ("m2_backbone_int8_pt", "M2 per-tensor"), ("m3_backbone_neck_int8_pc", "M3 per-channel"), ("m3_backbone_neck_int8_pt", "M3 per-tensor")):
    o, t, c = ST["matched_calibration_640px"][k]["onnx_percentile"], ST["matched_calibration_640px"][k]["tflite_minmax"], cm[k]["mAP50-95"]
    mm.append([lab, f"{o:.4f}", f"{c:.4f}", f"{t:.4f}", f"{(o - t) * 100:+.2f}", f"{(c - t) * 100:+.2f}"])
T_MM = table(["Design", "ONNX, percentile", "ONNX, min-max", "TFLite (min-max)", "ONNX pct. minus TFLite (pts)", "ONNX min-max minus TFLite (pts)"], mm,
             "Table 5. The same designs with the calibration method matched (640 px models, per-channel and per-tensor). Only the calibration method changes between columns 2 and 3.")

cpu = env["cpu"].replace("(R)", "").replace("(TM)", "")



def make_chart(out_dir):
    """Speed-up versus retained accuracy for every model, three runtimes. Returns the file name."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5))
    sets = [("ONNX Runtime", "#1f4e9e", [k for k, _ in groups["ORT"]], BL), ("OpenVINO", "#c0551b", [k for k, _ in groups["OpenVINO"]], L("ov_m0_fp32")),
            ("TFLite", "#1b7f5c", [k for k, _ in groups["TFLite"] if not k.endswith("_io8")], TFL_BL)]
    for name, col, keys, ref in sets:
        pts = [(ref / L(k), acc[ACC_KEY.get(k, k)]["mAP50-95"] / BASE * 100, k) for k in keys if k in lat and ACC_KEY.get(k, k) in acc]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=col, label=name)
        for x, y, k in pts:
            if k.endswith(("_pc", "_mixed")):
                d = "M1" if "m1" in k else "M2" if "m2" in k else "M3" if "m3" in k else ""
                ax.annotate(d, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(2, color="#999", ls="--", lw=1); ax.text(2.01, ax.get_ylim()[0] + 0.5, "2x", color="#666", fontsize=9)
    ax.axhline(95, color="#999", ls=":", lw=1); ax.text(0.99, 95.3, "95% retained", color="#666", fontsize=9)
    ax.set_xlabel("speed-up over the FP32 model of the same runtime"); ax.set_ylabel("accuracy retained (mAP50-95, % of FP32)")
    ax.yaxis.grid(True, color="#e6e6e6"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(Path(out_dir) / "8_speed_accuracy_tradeoff.png", dpi=150); plt.close(fig)
    return "8_speed_accuracy_tradeoff.png"


PP = J("postprocess.json")
try:
    import ai_edge_litert
    LITERT = ai_edge_litert.__version__
except ImportError:
    LITERT = "not installed"
_pick = lambda k: acc.get(k, {}).get("mAP50-95")
_rt_rows = [[lab, f"{RP[k]['tf']['median_ms']:.1f}", f"{RP[k]['litert']['median_ms']:.1f}", f"{RP[k]['tf']['median_ms'] / RP[k]['litert']['median_ms']:.2f}x"]
            for k, lab in (("s512_m0_fp32", "FP32 (no quantization)"), ("t512_m1_all_int8_pc", "M1 per-channel"), ("t512_m2_backbone_int8_pc", "M2 per-channel"),
                           ("t512_m3_backbone_neck_int8_pc", "M3 per-channel"), ("t512_m3_backbone_neck_int8_pt", "M3 per-tensor")) if k in RP]
RT_TABLE = table(["Same .tflite file (512 px)", "TensorFlow 2.15 interpreter (ms)", "LiteRT interpreter (ms)", "Speed-up from the runtime"], _rt_rows,
                 "Table 7. One image, batch 1, 4 pinned cores, 20 warm-up and 100 timed calls, model time only. The files and their accuracy are identical; only the program that runs them differs.")
def _e2e(label, model_ms, post_key, size):
    pre = PP["pre_ms"] * (size / 640) ** 2
    post = PP[post_key]
    return label, pre, model_ms, post, pre + model_ms + post
_rows = [_e2e("FP32 baseline (OpenVINO), 640 px, numpy post-processing, threshold 0.001", L("ov_m0_fp32"), "post_ms_numpy_conf0.001", 640),
         _e2e("FP32 baseline (OpenVINO), 640 px, numpy post-processing, threshold 0.25", L("ov_m0_fp32"), "post_ms_numpy_conf0.25", 640),
         _e2e("INT8 M3 (OpenVINO), 512 px, compiled post-processing, threshold 0.25", L("ovq_f512_m3_mixed"), "post_ms_compiled_conf0.25", 512),
         _e2e("INT8 M1 (TFLite), 512 px, compiled post-processing, threshold 0.25", L("tflite_t512_m1_all_int8_pc"), "post_ms_compiled_conf0.25", 512),
         _e2e("INT8 M2 (ONNX Runtime), 512 px, compiled post-processing, threshold 0.25", L("s512_m2_backbone_int8_pc"), "post_ms_compiled_conf0.25", 512)]
_ref = _rows[1][4]
E2E_TABLE = table(["Pipeline", "Pre (ms)", "Model (ms)", "Post (ms)", "Total (ms)", "vs FP32 baseline at threshold 0.25"],
                  [[r[0], f"{r[1]:.1f}", f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.1f}", f"{_ref / r[4]:.2f}x"] for r in _rows],
                  "End-to-end time per image on the benchmark CPU (4 pinned cores). Preprocessing at other sizes is scaled by pixel count; post-processing time is that measured at 640 px (a conservative figure for 512 px inputs).")

SECTION = f"""<h2>1. Measurement protocol</h2>
<div class="tablewrap"><table class="text"><caption>Table 1. How speed and memory were measured.</caption><tbody>
<tr><td>CPU</td><td>{html.escape(cpu)}: {env['physical_cores']} physical cores, {env['logical_processors']} logical processors; AVX-512 with VNNI (fast INT8 dot products).</td></tr>
<tr><td>Cores used</td><td>{proto['threads']} threads. {html.escape(proto['thread_placement'])}. Priority: {proto['process_priority']}.</td></tr>
<tr><td>Batch</td><td>{proto['batch_size']} image per call. Input 1x3x640x640 float32 for the FP32 reference models and 1x3x512x512 float32 for the INT8 models (preprocessing timed separately). Pre- and post-processing are timed separately and are not in the headline figure.</td></tr>
<tr><td>Warm-up</td><td>{proto['warmup_calls_per_round']} untimed calls before each timed measurement. The first call is slow (memory allocation, kernel selection); the time settles after 1 to 2 calls, so 20 is a wide margin.</td></tr>
<tr><td>Timed runs</td><td>{proto['timed_calls_per_round']} calls per measurement, cycling over 10 fixed images.</td></tr>
<tr><td>Rounds</td><td>{proto['rounds']} rounds. {html.escape(proto['round_order'])}. Headline = {html.escape(proto['statistic'])}. The round spread column is (max minus min) over the median.</td></tr>
<tr><td>Machine state</td><td>Idle apart from the benchmark, on AC power, Windows power plan: Balanced. All models were measured in the same session, interleaved.</td></tr>
<tr><td>Memory</td><td>Increase in resident memory over a process that already has the runtime loaded, sampled every 2 ms during inference, median of 3 runs.</td></tr>
<tr><td>Software</td><td>ONNX Runtime {env['packages_main_env']['onnxruntime']}, OpenVINO {html.escape(env['packages_main_env']['openvino'].split('-')[0])}, TensorFlow {env['packages_tensorflow_env']['tensorflow']} (used to convert models to TFLite), LiteRT {LITERT} (runs the TFLite models), Python {env['python']}.</td></tr>
</tbody></table></div>

<h2>2. Why ONNX and TFLite differ for the same design</h2>
<p>The FP32 baselines are identical (mAP50-95 {BASE:.4f}). Quantization is not: ONNX Runtime calibrates with percentile clipping, while the TFLite converter only offers min-max. Percentile ignores rare outliers and so gives finer steps for the typical values. Rebuilding the ONNX models with min-max calibration removes most of the gap (table below, measured on 640 px models): the remaining differences are in the range of {min(float(r[5]) for r in mm):+.2f} to {max(float(r[5]) for r in mm):+.2f} mAP points, while the percentile versions are {min(float(r[4]) for r in mm):+.2f} to {max(float(r[4]) for r in mm):+.2f} points above TFLite. The gap is a calibration effect, not a property of the format.</p>
{T_MM}

<h2>3. Speed and accuracy, every model</h2>
<p>All three runtimes, same protocol. The FP32 baseline takes {BL:.1f} ms per image with ONNX Runtime (batch 1, 4 pinned cores), {L('ov_m0_fp32') if 'ov_m0_fp32' in lat else float('nan'):.1f} ms with OpenVINO and {TFL_BL:.1f} ms with TFLite.</p>
{tables}

<h2>4. Why INT8 is not 2x or 4x faster here</h2>
<ul>
<li><b>The convolutions are only part of the time.</b> In FP32 the SiLU activation is fused into the convolution. In INT8 it is a separate sigmoid and multiply on every layer, and its cost grows with the size of the activation tensor, which is largest in the first layers.</li>
<li><b>Other operators do not speed up:</b> in ONNX Runtime the Resize and Concat steps of the neck need extra requantization work, so quantizing the neck (M3) is slower than leaving it in FP32 (M2). This was seen in the per-operation profile (<code>results/profile.json</code>).</li>
<li><b>Result with the smaller input size:</b> the INT8 models take a 512 px picture instead of 640 px, which removes about a third of the computation, and run {L('ov_m0_fp32') / L('ovq_f512_m3_mixed'):.2f}x (OpenVINO M3), {TFL_BL / L('tflite_t512_m1_all_int8_pc'):.2f}x (TFLite M1) and {BL / L('s512_m2_backbone_int8_pc'):.2f}x (ONNX Runtime M2) faster than the 640 px FP32 model of the same runtime, at 94 to 96% of its accuracy.</li>
<li>A 4x gain needs hardware where INT8 is not just faster but the only path (an NPU); this CPU cannot show that.</li>
</ul>

<h2>5. The input size</h2>
<p>Computation falls with the square of the input size, so the INT8 models take a 512 x 512 picture while the FP32 reference takes 640 x 640 (512 x 512 is 0.64 of 640 x 640). The same network at 512 px without quantization scores {acc['f512_fp32']['mAP50-95'] / BASE * 100:.1f}% of the 640 px FP32 accuracy, so the smaller picture itself costs about {100 - acc['f512_fp32']['mAP50-95'] / BASE * 100:.0f} points and quantization costs the rest. Small objects are the ones lost first, so a smaller input suits applications whose objects are not tiny.</p>

<h2>6. End-to-end time per image</h2>
{E2E_TABLE}
<p>The model is only part of the time per image. The reference post-processing (a numpy implementation of non-maximum suppression) took {PP['post_ms_numpy_conf0.001']:.1f} ms at the evaluation confidence threshold of 0.001 and {PP['post_ms_numpy_conf0.25']:.1f} ms at the usual deployment threshold of 0.25. Replacing it with a compiled non-maximum-suppression routine gives identical detections on {PP['identical_images_of_100']} of 100 test images and takes {PP['post_ms_compiled_conf0.001']:.1f} ms and {PP['post_ms_compiled_conf0.25']:.1f} ms. Preprocessing (letterbox and conversion) takes {PP['pre_ms']:.1f} ms.</p>

<h2>TFLite: how the models are made and run</h2>
<p>The TFLite models follow the standard TensorFlow Lite / LiteRT recipe for post-training integer quantization (see the sources below):</p>
<ul>
<li><b>Representative data.</b> A set of typical inputs, preprocessed exactly like at inference, gives the converter the range of every activation. Here: 500 train2017 pictures letterboxed to the model's input size (the LiteRT example uses 100).</li>
<li><b>Weights and activations.</b> Weights are int8, symmetric, per output channel in convolutions by default (the &ldquo;channel-wise&rdquo; models) or one scale per tensor (the &ldquo;layer-wise&rdquo; models, a converter switch); activations are int8, one scale per tensor. TFLite's converter offers only min-max range calibration, which is why it differs from the percentile calibration used for ONNX Runtime.</li>
<li><b>Keeping layers in float inside one file.</b> M2 and M3 keep part of the network in float. The converter's quantization debugger can exclude chosen operations (<code>denylisted_nodes</code>), so the whole model stays in <b>one .tflite file</b>. The converted graph has lost the layer names, so the excluded operations are found from the structure: everything that is not an ancestor of the backbone outputs (M2) or of the neck outputs (M3); the three boundary tensors are located by comparing values with separately converted parts. In M3 the 57 backbone and neck convolutions are INT8 and the 3 Detect-head convolutions are float. The documentation notes that on integer-only accelerators float layers fragment the model, which is why the all-INT8 M1 is the design to try first on an NPU.</li>
<li><b>Integer input and output.</b> Integer-only targets usually also need int8 model inputs and outputs. The rows marked &ldquo;integer input/output&rdquo; are M1 built this way (the program that runs them quantizes the picture and dequantizes the result with the model's own scales).</li>
<li><b>Check the FP32 conversion first.</b> The FP32 TFLite model scores the same as the FP32 ONNX model ({acc['tflite_m0_fp32']['mAP50-95']:.4f}), so conversion itself loses nothing and all differences come from quantization.</li>
</ul>
<p><b>The runtime mattered more than the model.</b> The TFLite models looked slow next to ONNX Runtime and OpenVINO, but the same files run much faster in the current LiteRT interpreter than in the one built into TensorFlow 2.15 (table below). INT8 TFLite models are now among the fastest in this report, and they use less memory too. All TFLite timings and memory figures in this report were measured with LiteRT {LITERT}; accuracy is unaffected by the choice of runtime (checked on 500 images for M3 per-channel: the same score with both). The gain is specific to INT8: the float model runs about the same in both interpreters.</p>
{RT_TABLE}
<p>Sources: <a href="https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization">LiteRT post-training quantization</a>, <a href="https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_debugger">quantization debugger</a> (selective quantization), <a href="https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec">8-bit quantization specification</a>.</p>

<h2>7. Limitations</h2>
<ul>
<li>All timings are from a laptop CPU; an NPU's ranking and speed-ups can differ. The exact target board is not known.</li>
<li>The TFLite and OpenVINO runtimes use different INT8 kernels from ONNX Runtime; the ranking between designs differs between runtimes (see the notes under the charts). LiteRT, ONNX Runtime and OpenVINO versions are listed in the protocol table.</li>
<li>Round-to-round spread on this laptop is a few percent for most models and occasionally much larger (an occasional round is an outlier); the median of nine rounds is robust to that, but differences of a few percent between INT8 variants are not meaningful, while differences to FP32 are.</li>
<li>OpenVINO INT8 models were quantized with NNCF (mixed preset, 500 calibration images); results for ONNX Runtime QDQ and OpenVINO are different quantizers and runtimes, not one model on two engines.</li>
</ul>
"""


def embed(img_b64, limits=True):
    """SECTION with the standalone numbering removed and the trade-off chart inserted (img_b64 = <img> tag)."""
    out = re.sub(r"<h2>\d+\. ", "<h2>", SECTION)
    out = re.sub(r"Table \d+\. ", "", out)
    if not limits:                      # the full report has its own Limitations section
        out = re.sub(r"<h2>(?:\d+\. )?Limitations</h2>.*?</ul>", "", out, flags=re.S)
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
