"""Build results/report.html: plain white, self-contained benchmark report (charts embedded). Reads results/*.json."""
import base64
import html
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_sections  # noqa: E402
import eras  # noqa: E402
extra_sections.make_chart(R / "charts")
J = lambda n: json.loads((R / n).read_text())
rows = J("summary.json")
ST = J("study_inputs.json")                                  # mAP50-95 of the 640 px models used by the box-scale and calibration studies
S = {(r["format"], r["model"]): r for r in rows}
paired, sens = J("paired.json"), J("sensitivity.json")
calib = J("calib_size.json") if (R / "calib_size.json").exists() else {}
F = ("ONNX", "m0_fp32")
base = S[F]
b_map = base["mAP50-95"]


def img(name, alt):
    return f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode((R / "charts" / name).read_bytes()).decode()}">'


def g(fmt, m):
    return S[(fmt, m)]


def pct(a, b):
    return f"{a / b * 100:.1f}%"


LABEL = {"m0_fp32": "FP32 baseline", "m1_all_int8_pc": "M1 all INT8, per-channel (pixel boxes)", "m1_all_int8_pt": "M1 all INT8, per-tensor (pixel boxes)",
         "m1_all_int8_pc_norm": "M1 all INT8, per-channel", "m1_all_int8_pt_norm": "M1 all INT8, per-tensor",
         "m2_backbone_int8_pc": "M2 backbone INT8, per-channel", "m2_backbone_int8_pt": "M2 backbone INT8, per-tensor",
         "m3_backbone_neck_int8_pc": "M3 backbone+neck INT8, per-channel", "m3_backbone_neck_int8_pt": "M3 backbone+neck INT8, per-tensor"}
def name(r):
    k = r["model"].replace("tflite_", "")
    if k.startswith(("s512_", "t512_")):                           # current stage: for ONNX the M1 file is the normalised-box model
        k = k.replace("s512_", "").replace("t512_", "") + ("_norm" if k.startswith("s512_m1") else "")
    return LABEL[k]
ONNX = [r for r in rows if r["format"] == "ONNX" and (r["model"] == "m0_fp32" or r["model"].startswith("s512_"))]            # current stage (512 px) + the 640 px FP32 baseline
TFL = [r for r in rows if r["format"] == "TFLite" and (r["model"] == "tflite_m0_fp32" or r["model"].startswith("tflite_t512_"))]


def table(cols, data, caption, num=True):
    th = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in data)
    return f'<div class="tablewrap"><table><caption>{caption}</caption><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


# ---- tables
t_acc = []
for r in [x for x in ONNX if "m1_all_int8_p" in x["model"] and not x["model"].endswith("_norm")]:
    pass
_ord = lambda r: (r["model"].replace("tflite_", "").replace("s512_", "").replace("t512_", "")[:2], r["model"].endswith("_pt"))
acc_rows = sorted([r for r in ONNX if r["model"] not in ("m1_all_int8_pc", "m1_all_int8_pt")], key=_ord) + sorted([r for r in TFL], key=_ord)
for r in acc_rows:
    t_acc.append([r["format"], name(r), f"{r['mAP50-95']:.4f}", f"{r['accuracy_retained_pct']:.1f}%", f"{r['mAP50']:.4f}",
                  f"{r['AP_small']:.3f}", f"{r['AP_medium']:.3f}", f"{r['AP_large']:.3f}"])
T_ACC = table(["Format", "Model", "mAP50-95", "Retained", "mAP50", "AP small", "AP medium", "AP large"], t_acc,
              "Table 2. Accuracy on all 5000 COCO val2017 images. &ldquo;Retained&rdquo; = mAP50-95 as a share of the FP32 model's.")

t_prf = [[r["format"], name(r), f"{r['precision@0.25']:.3f}", f"{r['recall@0.25']:.3f}", f"{r['F1@0.25']:.3f}", f"{r['best_F1']:.3f} @ {r['best_F1_conf']:.2f}"] for r in acc_rows]
T_PRF = table(["Format", "Model", "Precision", "Recall", "F1", "Best F1 @ conf"], t_prf,
              "Table 3. Precision, recall and F1 at confidence 0.25, matched at IoU 0.5 with the official COCO matching rules (crowd regions ignored), micro-averaged over all classes and all 5000 images.")

t_size = [[r["format"], name(r), f"{r['size_mb']:.2f}", f"{r['compression_ratio']:.2f}x"] for r in acc_rows]
T_SIZE = table(["Format", "Model", "Size (MB)", "Compression"], t_size,
               "Table 4. File size (all stages of a pipeline summed) and compression ratio versus the FP32 model of the same format.")

N_R = J("latency.json")["m0_fp32"]["rounds"]
t_rt = [[r["format"], name(r), f"{r['latency_ms_median']:.1f}", f"{r['latency_ms_p95']:.1f}", f"{r['speedup_vs_fp32']:.2f}x",
         f"{r['mem_load_mb']:.1f}", f"{r['mem_peak_added_mb']:.0f}"] for r in acc_rows]
T_RT = table(["Format", "Model", "Median ms", "p95 ms", "Speed-up", "Memory added at load (MB)", "Peak memory added (MB)"], t_rt,
             f"Table 5. Latency (batch 1, 4 threads; median of {N_R} interleaved rounds of 100 runs after warm-up) and memory (increase in the process's resident memory over a baseline with the runtime already loaded; fresh process per model).")

t_pair = [[p["comparison"], f"{p['diff'] * 100:+.2f}", f"[{p['ci95'][0] * 100:+.2f}, {p['ci95'][1] * 100:+.2f}]", "yes" if p["significant"] else "no"] for p in paired]
T_PAIR = table(["Comparison (A minus B)", "Difference (mAP points)", "95% interval", "Real difference?"], t_pair,
               "Table 6. Paired comparisons over 10 disjoint image folds. A difference is real if the interval excludes zero.")

cal = {int(k): v["mAP50-95"] for k, v in calib.items()}
cal[500] = ST["m3_pc_onnx"]
T_CAL = table(["Calibration images", "mAP50-95", "vs 500 images"], [[str(n), f"{cal[n]:.4f}", f"{(cal[n] - cal[500]) * 100:+.2f} pts"] for n in sorted(cal)],
              "Table 7. M3 per-channel (ONNX) calibrated on different numbers of train2017 images, scored on all 5000 val images.")

# ---- precision / recall / F1 notes (data-driven)
fp = g("ONNX", "m0_fp32")
d = lambda r, k: r[k] - fp[k]
hf = [r for r in acc_rows if r["model"].replace("tflite_", "").replace("s512_", "").replace("t512_", "")[:2] in ("m2", "m3")]
m1r = [r for r in acc_rows if r["model"].replace("tflite_", "").replace("s512_", "").replace("t512_", "")[:2] == "m1"]
rg2 = lambda rs, k: (min(d(r, k) for r in rs), max(d(r, k) for r in rs))
pP, pR, pF, qP, qR, qF = rg2(hf, "precision@0.25"), rg2(hf, "recall@0.25"), rg2(hf, "F1@0.25"), rg2(m1r, "precision@0.25"), rg2(m1r, "recall@0.25"), rg2(m1r, "F1@0.25")
sg = lambda x: "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"
prf_text = (f"Relative to FP32, designs with an FP32 head (M2, M3) change precision by {sg(pP[0])} to {sg(pP[1])}, recall by {sg(pR[0])} to {sg(pR[1])} and F1 by {sg(pF[0])} to {sg(pF[1])}; "
            f"the all-INT8 designs (M1) change precision by {sg(qP[0])} to {sg(qP[1])}, recall by {sg(qR[0])} to {sg(qR[1])} and F1 by {sg(qF[0])} to {sg(qF[1])}. "
            f"Recall falls in every quantized model, while precision is roughly unchanged or slightly higher when the head stays FP32 and lower when it does not.")

# ---- numbers used in the text
m2o, m3o, m1n = g("ONNX", "s512_m2_backbone_int8_pc"), g("ONNX", "s512_m3_backbone_neck_int8_pc"), g("ONNX", "s512_m1_all_int8_pc")
m2t, m3t, m1t = g("TFLite", "tflite_t512_m2_backbone_int8_pc"), g("TFLite", "tflite_t512_m3_backbone_neck_int8_pc"), g("TFLite", "tflite_t512_m1_all_int8_pc")
m1n_d2 = {"mAP50-95": ST["m1_pc_norm"]}                                      # earlier (640 px) all-INT8 model, used by the box-scale study
sb = sens["fp32"]
lay = sorted(((int(k[1:]), (sb - v) / sb * 100) for k, v in sens.items() if k != "fp32"), key=lambda x: -x[1])
head_loss, l23 = dict(lay)[24], dict(lay)[23]
qdq = J("sensitivity_qdq.json")
not_q = [int(k[1:]) for k, v in qdq.items() if k.startswith("L") and not k.endswith("_total") and v == 0]
nq_txt = " and ".join(str(i) for i in not_q)
others = max(v for i, v in lay if i != 24)
pcpt = [p for p in paired if "per-channel minus per-tensor" in p["comparison"]]
pc_lo, pc_hi = min(p["diff"] for p in pcpt) * 100, max(p["diff"] for p in pcpt) * 100
head_pairs = [p for p in paired if "head FP32" in p["comparison"]]
h_lo, h_hi = min(p["diff"] for p in head_pairs) * 100, max(p["diff"] for p in head_pairs) * 100
onnx_vs_tfl = [p for p in paired if "ONNX minus TFLite" in p["comparison"]]
ot_lo, ot_hi = min(p["diff"] for p in onnx_vs_tfl) * 100, max(p["diff"] for p in onnx_vs_tfl) * 100
_cg = [(g("ONNX", "s512_" + d)["mAP50-95"] - g("TFLite", "tflite_t512_" + d)["mAP50-95"]) * 100 for d in ("m1_all_int8_pc", "m1_all_int8_pt", "m2_backbone_int8_pc", "m2_backbone_int8_pt", "m3_backbone_neck_int8_pc", "m3_backbone_neck_int8_pt")]
cur_gap_lo, cur_gap_hi = min(_cg), max(_cg)
less = lambda a, b: (1 - a / b) * 100
o_fp, t_fp = g("ONNX", "m0_fp32"), g("TFLite", "tflite_m0_fp32")
o_int = [r for r in ONNX if r["model"] not in ("m0_fp32", "m1_all_int8_pc", "m1_all_int8_pt")]
t_m1 = [r for r in TFL if "m1_" in r["model"]]
t_m23 = [r for r in TFL if "m2_" in r["model"] or "m3_" in r["model"]]
rng = lambda rs, key, ref: (min(less(r[key], ref[key]) for r in rs), max(less(r[key], ref[key]) for r in rs))
rr = lambda a: f"{a[0]:.0f}%" if round(a[0]) == round(a[1]) else f"{a[0]:.0f} to {a[1]:.0f}%"      # "66%" or "35 to 42%"
sp_o, sp_t1, sp_t23 = (rng(o_int, "latency_ms_median", o_fp), rng(t_m1, "latency_ms_median", t_fp), rng(t_m23, "latency_ms_median", t_fp))
spread = max(r["latency_round_spread_pct"] for r in rows if r.get("latency_round_spread_pct") is not None)
o_m1 = [r for r in ONNX if "m1_" in r["model"]]
o_m23 = [r for r in ONNX if "m2_" in r["model"] or "m3_" in r["model"]]
mp = lambda rs: (min(r["mem_peak_added_mb"] for r in rs), max(r["mem_peak_added_mb"] for r in rs))
mchg = lambda rs, ref: (min(r["mem_peak_added_mb"] / ref["mem_peak_added_mb"] * 100 - 100 for r in rs), max(r["mem_peak_added_mb"] / ref["mem_peak_added_mb"] * 100 - 100 for r in rs))
mo1, mo23, mt1, mt23 = mchg(o_m1, o_fp), mchg(o_m23, o_fp), mchg(t_m1, t_fp), mchg(t_m23, t_fp)
vs2 = lambda a: (f"{fmtr(a)} less" if a[1] < 0 else f"{fmtr(a)} more" if a[0] > 0 else f"between {a[0]:+.0f}% and {a[1]:+.0f}%")


def fmtr(a):
    lo, hi = sorted((abs(a[0]), abs(a[1])))
    return f"{lo:.0f} to {hi:.0f}%" if round(lo) != round(hi) else f"{lo:.0f}%"

# ---- latency evidence, calibration-method evidence, profile evidence (data-driven)
latj = J("latency.json")
n_rounds = latj["m0_fp32"]["rounds"]
ONNX_KEYS = ["s512_m1_all_int8_pc", "s512_m1_all_int8_pt", "s512_m2_backbone_int8_pc", "s512_m2_backbone_int8_pt", "s512_m3_backbone_neck_int8_pc", "s512_m3_backbone_neck_int8_pt"]
TFL_KEYS = ["tflite_t512_m1_all_int8_pc", "tflite_t512_m1_all_int8_pt", "tflite_t512_m2_backbone_int8_pc", "tflite_t512_m2_backbone_int8_pt", "tflite_t512_m3_backbone_neck_int8_pc", "tflite_t512_m3_backbone_neck_int8_pt"]
faster_rounds = lambda keys, base: [sum(a < b for a, b in zip(latj[k]["round_medians_ms"], latj[base]["round_medians_ms"])) for k in keys]
fr_o, fr_t = faster_rounds(ONNX_KEYS, "m0_fp32"), faster_rounds(TFL_KEYS, "tflite_m0_fp32")
spread = max(latj[k]["round_spread_pct"] for k in ONNX_KEYS + TFL_KEYS)
if (R / "calib_method.json").exists():
    cm = J("calib_method.json")["minmax"]["mAP50-95"]
    pe, tf_ = ST["m3_pc_onnx"], ST["m3_pc_tflite"]      # earlier (640 px) models, as in the calibration study
    cm_text = (f"To test the calibration explanation directly (earlier 640 px models), the ONNX M3 per-channel model was rebuilt with min-max calibration (the method the TFLite converter uses): it scores {cm:.4f}, against {pe:.4f} with percentile calibration and {tf_:.4f} for the TFLite model. "
               + (f"Min-max calibration therefore reproduces the TFLite accuracy (difference {abs(cm - tf_) * 100:.2f} mAP points), so the calibration method accounts for essentially all of the ONNX-to-TFLite gap for this design." if abs((pe - cm) / (pe - tf_) - 1) < 0.1 else f"Switching to min-max therefore accounts for {(pe - cm) / (pe - tf_) * 100:.0f}% of the ONNX-to-TFLite gap for this design."))
else:
    cm_text = "A likely reason is the calibration method (percentile in ONNX Runtime, min-max in the TFLite converter); this was not isolated."
prof_text = ""
if (R / "profile.json").exists():
    pj = J("profile.json")
    op = lambda m, k: pj[m]["by_op_ms"].get(k, 0.0)
    rg_ = lambda m, k: pj[m]["by_region_ms"].get(k, 0.0)
    act = "activation (sigmoid/mul)"
    prof_text = (f" A per-operation profile of the earlier 640 px ONNX models explains why the gain is not larger: INT8 convolutions are faster ({op('M1 per-channel', 'conv'):.1f} ms against {op('FP32', 'conv'):.1f} ms for FP32), "
                 f"but the SiLU activation (a sigmoid followed by a multiply) costs {op('M1 per-channel', act):.1f} ms in the INT8 model against {op('FP32', act):.1f} ms in FP32, where it is fused into the convolution, so most of the convolution saving is spent on activations. "
                 f"Quantizing the neck does not speed it up: the neck region takes {rg_('M3 per-channel', 'neck'):.1f} ms in M3 and {rg_('M1 per-channel', 'neck'):.1f} ms in M1 against {rg_('FP32', 'neck'):.1f} ms in FP32, which is why M2 (neck left in FP32) is the fastest ONNX design. "
                 f"(Profiler kernel times include profiling overhead and do not add up to wall-clock time; this is how ONNX Runtime on this CPU behaves and says nothing about an NPU.)")

era_names = eras.era_chart_files(R / "charts")
html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLOv5n INT8 Benchmark</title>
<style>
:root {{ color-scheme: light; }} * {{ box-sizing: border-box; }}
html {{ background:#fff; }}
body {{ margin:0; color:#1a1a1a; background:#fff; font:17px/1.65 Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif; }}
.page {{ max-width:860px; margin:0 auto; padding:64px 24px 96px; }}
h1,h2,h3,.meta,table,figcaption {{ font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }}
h1 {{ font-size:30px; line-height:1.25; margin:0 0 10px; letter-spacing:-0.01em; }}
.meta {{ color:#666; font-size:14px; margin:0 0 36px; padding-bottom:20px; border-bottom:1px solid #ddd; }}
h2 {{ font-size:20px; margin:52px 0 12px; }} h3 {{ font-size:15px; margin:26px 0 6px; }}
p {{ margin:0 0 14px; }} ul {{ padding-left:22px; margin:0 0 14px; }} li {{ margin:6px 0; }}
code {{ font:0.88em Consolas,Menlo,monospace; background:#f5f5f5; padding:1px 5px; border-radius:3px; }}
.summary {{ border-top:2px solid #1a1a1a; border-bottom:1px solid #ddd; padding:18px 0 6px; }}
.tablewrap {{ overflow-x:auto; margin:22px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
caption {{ text-align:left; font-weight:700; padding:0 0 8px; font-size:13.5px; }}
th,td {{ padding:6px 10px 6px 0; text-align:right; font-variant-numeric:tabular-nums; vertical-align:top; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
thead th {{ border-top:1.5px solid #1a1a1a; border-bottom:1px solid #1a1a1a; font-weight:600; }}
tbody tr {{ border-bottom:1px solid #eee; }} tbody tr:last-child {{ border-bottom:1.5px solid #1a1a1a; }}
table.text td, table.text th {{ text-align:left; white-space:normal; }}
figure {{ margin:28px 0; }} figure img {{ display:block; width:100%; height:auto; }}
figcaption {{ font-size:13.5px; color:#444; margin-top:10px; line-height:1.5; }} figcaption b {{ color:#1a1a1a; }}
dl {{ margin:0; }} dt {{ font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif; font-weight:700; font-size:14.5px; margin-top:14px; }} dd {{ margin:2px 0 0; color:#333; }}
footer {{ margin-top:64px; padding-top:16px; border-top:1px solid #ddd; color:#777; font:13px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }}
@media print {{ .page {{ padding:0; max-width:none; }} h2 {{ break-after:avoid; }} figure,table {{ break-inside:avoid; }} }}
</style></head><body><main class="page">

<h1>INT8 Quantization of YOLOv5n for an Embedded NPU Target</h1>
<p class="meta">Benchmark report &nbsp;|&nbsp; YOLOv5n v7.0 (original, anchor-based) &nbsp;|&nbsp; COCO val2017, all 5000 images &nbsp;|&nbsp; ONNX Runtime, TFLite and OpenVINO, CPU, 4 pinned cores</p>

<div class="summary">
<p><b>Summary.</b> We compared an FP32 YOLOv5n baseline (mAP50-95 {b_map:.3f}, matching the published figure) with three INT8 designs, each quantized per-channel and per-tensor, in both ONNX and TFLite. The designs differ in which parts of the network stay in FP32: <b>M1</b> none, <b>M2</b> neck and head, <b>M3</b> head only.</p>
<ul>
<li><b>Best accuracy:</b> M2 per-channel keeps {m2o['accuracy_retained_pct']:.1f}% of FP32 accuracy in ONNX ({m2t['accuracy_retained_pct']:.1f}% in TFLite) at {m2o['compression_ratio']:.1f}x compression.</li>
<li><b>Best trade-off:</b> M3 per-channel keeps {m3o['accuracy_retained_pct']:.1f}% (ONNX) and {m3t['accuracy_retained_pct']:.1f}% (TFLite) at {m3o['compression_ratio']:.1f}x and {m3t['compression_ratio']:.1f}x compression ({m3o['size_mb']:.2f} MB and {m3t['size_mb']:.2f} MB).</li>
<li><b>Per-channel beats per-tensor</b> in every pairing, by {pc_lo:.1f} to {pc_hi:.1f} mAP points in the earlier (640 px) measurement, where the difference is statistically real in all of them.</li>
<li><b>The detection head is the sensitive part.</b> Quantizing the head alone costs {head_loss:.1f}% accuracy; every other single layer costs {others:.1f}% or less. Keeping the head in FP32 is worth {h_lo:.1f} to {h_hi:.1f} mAP points.</li>
<li><b>A fully INT8 model works</b> once box coordinates are normalised to 0-1 ({m1n['accuracy_retained_pct']:.1f}% retained in ONNX, {m1t['accuracy_retained_pct']:.1f}% in TFLite). With pixel-scale boxes it scores zero.</li>
<li><b>Speed and memory on this CPU:</b> INT8 ONNX models take {rr(sp_o)} less time per image than FP32 and all-INT8 TFLite models {rr(sp_t1)} less; TFLite pipelines that keep an FP32 stage {rr(sp_t23)} less (all against the 640 px FP32 baseline; the INT8 models run at 512 px). The all-INT8 TFLite pipelines also use about {fmtr(mt1)} less memory. These are desktop-CPU figures and do not transfer to an NPU (Section 7).</li>
<li><b>Input size:</b> the INT8 models take a 512 x 512 picture instead of 640 x 640, which removes about a third of the computation. At 512 px the OpenVINO M3 model is {extra_sections.L('ov_m0_fp32') / extra_sections.L('ovq_f512_m3_mixed'):.2f}x faster than the 640 px FP32 model at {extra_sections.acc['ovq_f512_m3_mixed']['mAP50-95'] / extra_sections.BASE * 100:.1f}% of its accuracy. See the sections after the six benchmarks.</li>
</ul></div>

{eras.eras_html(img, era_names)}
<h3>Reading the charts</h3>
{"".join(eras.era_note(m) for m in ("map50", "map", "size", "time", "mem", "f1"))}

<h2>1. Setup</h2>
<p><b>Stage shown.</b> Tables 2 to 5 and Figures 1, 4, 5 and 6 show the models: the INT8 models built at 512 x 512 input, with the 640 x 640 FP32 model as the baseline they replace. The studies in Sections 3, 4, 8 and 9 (box scale, layer sensitivity, calibration size, statistics) were done on the earlier 640 x 640 models and are labelled as such.</p>
<p><b>Models.</b> YOLOv5n has a backbone (layers 0-9), a PANet neck (10-23) and a Detect head (24). Table 1 lists the designs. Each INT8 design is built twice: <i>per-channel</i> (one scale per output channel of each convolution) and <i>per-tensor</i> (one scale per layer).</p>
<div class="tablewrap"><table class="text"><caption>Table 1. The designs compared.</caption>
<thead><tr><th>ID</th><th>Backbone</th><th>Neck (PANet)</th><th>Head (Detect)</th><th>Variants</th></tr></thead><tbody>
<tr><td>M0</td><td>FP32</td><td>FP32</td><td>FP32</td><td>baseline</td></tr>
<tr><td>M1</td><td>INT8</td><td>INT8</td><td>INT8</td><td>per-channel, per-tensor</td></tr>
<tr><td>M2</td><td>INT8</td><td>FP32</td><td>FP32</td><td>per-channel, per-tensor</td></tr>
<tr><td>M3</td><td>INT8</td><td>INT8</td><td>FP32</td><td>per-channel, per-tensor</td></tr></tbody></table></div>
<p><b>Quantization.</b> Static post-training INT8: weights signed 8-bit, activations 8-bit. In ONNX, ONNX Runtime QDQ with percentile calibration, keeping FP32 layers out by node name. In TFLite the converter uses full-integer conversion with min-max calibration. The current 512 px models are single files in which the layers that must stay FP32 (M2, M3) are excluded with the converter's quantization debugger; the earlier 640 px M2 and M3 were instead split at the backbone/neck/head boundaries into separate files chained by a small runner (see the section on how the TFLite models are made and run, below). TFLite models run on the LiteRT runtime. All variants calibrate on the same 500 images from COCO <i>train2017</i> (letterboxed to the input size of the model). OpenVINO models use NNCF post-training quantization (mixed preset) with the same images.</p>
<p><b>Evaluation.</b> All 5000 COCO val2017 images, scored with the official <code>pycocotools</code>, with preprocessing and post-processing (square letterbox at the model's input size (640 or 512), confidence 0.001, NMS IoU 0.6, up to 300 detections) matching the official YOLOv5 validation. The FP32 model scores {b_map:.3f}, in line with the published 28.0.</p>
<p><b>Statistics.</b> To tell real differences from noise, the 5000 images are split into 10 disjoint folds; each model is scored per fold and paired differences get a 95% t-interval (Table 6).</p>

<h2>2. Accuracy</h2>
<figure>{img('1_accuracy.png', 'Accuracy by design, format and granularity')}
<figcaption><b>Figure 1.</b> mAP50-95 for every design, granularity and format (higher is better). The dashed line is the FP32 baseline. Per-channel is above per-tensor in every group, and ONNX is above TFLite.</figcaption></figure>
{T_ACC}
<p><b>Per-channel versus per-tensor.</b> In all six pairings per-channel is better, by {pc_lo:.1f} to {pc_hi:.1f} mAP points (Table 6). Convolution output channels in this network have quite different weight ranges, and one shared scale wastes resolution on the narrow ones.</p>
<p><b>How much to quantize.</b> Moving from M2 to M3 (also quantizing the neck) costs a further {(m2o['mAP50-95'] - m3o['mAP50-95']) * 100:.1f} mAP points in ONNX per-channel, while shrinking the file from {m2o['size_mb']:.2f} to {m3o['size_mb']:.2f} MB. That is a good price for the saving.</p>
<p><b>ONNX versus TFLite.</b> ONNX is {cur_gap_lo:.1f} to {cur_gap_hi:.1f} mAP points better than TFLite for the same design (earlier 640 px models: {ot_lo:.1f} to {ot_hi:.1f}). {cm_text}</p>
<p><b>Object size.</b> Small objects have the lowest AP in every model. When the head is also INT8 (M1) they are hit hardest: 20 to 30% of small-object AP is lost, against 5 to 11% for large objects. With an FP32 head (M2, M3) the loss is spread across sizes, and no size class is consistently worst.</p>

<h2>3. The box-scale finding</h2>
<figure>{img('5_box_scale_fix.png', 'All-INT8 model with pixel versus normalised boxes')}
<figcaption><b>Figure 2.</b> The all-INT8 model (M1, ONNX). Left bar in each pair: boxes as pixels (0 to 640). Right bar: boxes divided by 640 (0 to 1). The dashed line is FP32.</figcaption></figure>
<p>The detector outputs one table holding box coordinates and confidence scores. With boxes up to 640 and scores below 1 sharing one INT8 scale, each INT8 step is about 2.5 wide, so every score rounds to zero and nothing is detected. Dividing the box output by 640 (done by changing the decode constants, with no added operations, and exactly reversible: the normalised FP32 model reproduces the original to within 6e-5 pixels) puts both on a comparable scale. M1 then recovers from 0.000 to {m1n_d2['mAP50-95']:.3f} (ONNX per-channel, 640 px). The official YOLOv5 TFLite export does the same, which is why the TFLite models use it throughout.</p>
<p>Even so, M1 remains below M3 by {h_lo:.1f} to {h_hi:.1f} mAP points, so the head is still worth protecting if the deployment allows an FP32 stage.</p>

<h2>4. Layer sensitivity</h2>
<figure>{img('6_layer_sensitivity.png', 'Accuracy lost per quantized layer')}
<figcaption><b>Figure 3.</b> Accuracy lost when only one layer is INT8 and everything else stays FP32 (ONNX, 0-1 boxes, first 1000 val images, FP32 reference {sb:.4f}). The grey band marks differences smaller than 0.3%, which are noise. Hatched bars: layers that were not actually quantized in isolation.</figcaption></figure>
<p>The Detect head (layer 24) costs {head_loss:.1f}%, far more than any other layer that was actually quantized. The next largest are layer 23, the last neck block feeding the head ({l23:.1f}%), and layer 2 ({dict(lay)[2]:.1f}%). <b>Caveat:</b> in this test, layers {nq_txt} (the upsample layers, which hold no weights) received no quantization at all when isolated, because ONNX Runtime inserts no quantize/dequantize operations around them on their own. Their zero is therefore not a result, and they are drawn hatched in Figure 3. All other layers were confirmed to contain quantization operations. The study supports protecting the output end of the network; it does not show that any particular layer is free to quantize.</p>

<h2>5. Size and compression</h2>
<figure>{img('2_size_vs_accuracy.png', 'Size versus accuracy')}
<figcaption><b>Figure 4.</b> File size against accuracy for every model. The best models sit toward the upper right (small and accurate). The cost of each extra INT8 stage is visible as a step left and down.</figcaption></figure>
{T_SIZE}
<p>M1 compresses {m1n['compression_ratio']:.1f}x, M3 {m3o['compression_ratio']:.1f}x and M2 {m2o['compression_ratio']:.1f}x. M2 and M3 stay larger than M1 because the FP32 stages are stored in full precision.</p>

<h2>6. Precision, recall and F1</h2>
<figure>{img('4_precision_recall_f1.png', 'Precision, recall and F1')}
<figcaption><b>Figure 5.</b> Precision, recall and F1 at confidence 0.25 for FP32 and the three ONNX per-channel designs.</figcaption></figure>
{T_PRF}
<p>{prf_text}</p>

<h2>7. Speed and memory</h2>
<figure>{img('3_latency_memory.png', 'Latency and memory')}
<figcaption><b>Figure 6.</b> Median time per image (whisker to the 95th percentile) and memory added to the process during inference, for every model.</figcaption></figure>
{T_RT}
<p><b>Speed.</b> (All percentages below are against the 640 px FP32 model; part of the saving comes from the smaller 512 px picture the INT8 models take. The charts and notes below give FP32 at 512 px as well, which isolates the effect of INT8.) Quantization is working: ONNX Runtime executes the INT8 designs with integer convolution kernels. Each model was timed over {n_rounds} interleaved rounds (fresh process per round). On this CPU, INT8 ONNX models take {rr(sp_o)} less time per image than FP32, the all-INT8 TFLite models {rr(sp_t1)} less, and the TFLite designs that keep an FP32 stage (M2, M3) {rr(sp_t23)} less, because those layers still run as float. Each ONNX INT8 model was faster than the FP32 model in {min(fr_o)} of the {n_rounds} rounds, and each TFLite INT8 pipeline in {min(fr_t)} of {n_rounds}. Timing is noisy, though: the same model varied by up to {spread:.0f}% between rounds (laptop CPU frequency and temperature drift), so small gaps <i>between INT8 variants</i> (for example per-channel versus per-tensor) are not meaningful, while the gap to FP32 is.{prof_text} <b>None of this is NPU speed.</b> An NPU executes only INT8 operations, so on a real board the FP32 stages of M2 and M3 would fall back to the host CPU, and the ranking could differ substantially. That needs measuring on the target hardware.</p>
<p><b>Memory.</b> (All percentages below are against the 640 px FP32 model; part of the saving comes from the smaller 512 px picture the INT8 models take. The charts and notes below give FP32 at 512 px as well, which isolates the effect of INT8.) Table 5 gives the memory a model adds to a process that already has the runtime loaded, at load time and at its peak during inference (which includes the activation tensors). The FP32 ONNX model adds {o_fp["mem_peak_added_mb"]:.0f} MB at peak; the all-INT8 ONNX models add {vs2(mo1)}. The ONNX models that keep FP32 parts (M2, M3) add {vs2(mo23)}: they save less than the all-INT8 models because the FP32 parts keep full-size tensors (a likely explanation, not verified). In TFLite the effect is clearer: the FP32 pipeline adds {t_fp["mem_peak_added_mb"]:.0f} MB, the all-INT8 pipelines {vs2(mt1)}, and M2 and M3 {vs2(mt23)}. Memory added at load is similar across ONNX models (about {min(r["mem_load_mb"] for r in ONNX if r["mem_load_mb"] is not None):.0f} to {max(r["mem_load_mb"] for r in ONNX if r["mem_load_mb"] is not None):.0f} MB), probably because the runtime re-packs weights whatever their precision (not verified). The runtime's own baseline (about {o_fp["mem_baseline_mb"]:.0f} MB for ONNX Runtime and {t_fp["mem_baseline_mb"]:.0f} MB for TensorFlow) is excluded. On a memory-limited board, activation memory matters more than the file size.</p>

<h2>8. Calibration set size</h2>
<figure>{img('7_calibration_size.png', 'Accuracy versus number of calibration images') if (R / 'charts' / '7_calibration_size.png').exists() else ''}
<figcaption><b>Figure 7.</b> M3 per-channel (ONNX) accuracy against the number of calibration images.</figcaption></figure>
{T_CAL}
<p>The spread across calibration sizes is {(max(cal.values()) - min(cal.values())) * 100:.2f} mAP points, so for this model a few dozen images are nearly as good as 500. Calibration size is not where accuracy is won or lost; granularity and head handling matter far more.</p>

<h2>9. Statistical comparison</h2>
<p>With 5000 images every INT8 model is measurably worse than FP32. The question that matters for choosing between variants is whether they differ from each other, which Table 6 answers.</p>
{T_PAIR}
<p>Differences in this table are measured in mAP points on 500-image folds, so they are slightly larger in absolute terms than the same differences on the full set, but they rank the models the same way.</p>

{extra_sections.embed(img("8_speed_accuracy_tradeoff.png", "Speed-up versus accuracy retained"), limits=False)}

<h2>10. Recommendations</h2>
<ul>
<li>Use <b>per-channel</b> quantization wherever the target runtime supports it.</li>
<li>Normalise the box output to 0-1 (divide by the input size) before quantizing any model that includes the head.</li>
<li>For accuracy first, choose <b>M2</b>; for the best size-accuracy balance, choose <b>M3</b>; choose <b>M1</b> only if the whole model must run as INT8 (for example entirely on an NPU), accepting the accuracy cost.</li>
<li>If speed matters more than the last few points of accuracy, <b>lower the input size</b>: at 512 px the M2 and M3 models keep 93 to 97% of the FP32 accuracy in all three runtimes and run 1.7 to 2.2 times as fast as the 640 px FP32 model of the same runtime. Check that the objects of interest are not tiny.</li>
<li>Use a compiled non-maximum-suppression routine; the reference numpy one took longer than the model.</li>
<li>Do not spend effort on calibration set size beyond about 100 images for this model (earlier 640 px study).</li>
<li>Next: time the candidates on the target board; try channel pruning and quantization-aware training; try a mixed design that keeps only the most sensitive layers (23 and 24) in FP32.</li>
</ul>

<h2>11. Limitations</h2>
<ul>
<li><b>No NPU measurements.</b> Timing is from a laptop CPU (ONNX Runtime, TFLite with XNNPACK and OpenVINO). It is a proxy only, and the ranking of designs differs between runtimes (see the notes under the charts); the exact target board is not known.</li>
<li><b>Different calibration methods</b> in ONNX (percentile) and TFLite (min-max) mean the two formats are not like-for-like; the TFLite converter offers no percentile option. OpenVINO models use a third quantizer (NNCF), so ONNX Runtime and OpenVINO are different quantizers and runtimes, not one model on two engines.</li>
<li><b>Memory</b> is the increase in resident memory of a desktop process (sampled every 2 ms), not NPU or device memory, and it excludes the runtime's own baseline. OpenVINO's INT8 models use more than its FP32 model; the reason (probably compiling the quantized graph) was not verified.</li>
<li><b>Timing noise:</b> round-to-round spread is a few percent for most models and occasionally much larger; the median of nine interleaved rounds is used. Differences of a few percent between INT8 variants are not meaningful.</li>
<li><b>Layer sensitivity</b> uses the first 1000 images and one layer at a time; interactions between layers are not captured. The sensitivity, calibration-size, box-scale and statistical studies were run on the earlier 640 px models.</li>
<li><b>Data.</b> COCO val2017 only. Results on other image domains may differ.</li>
</ul>

<h2>Appendix. Terms</h2>
<dl>
<dt>mAP50-95 / mAP50</dt><dd>The standard detector accuracy score (0 to 1), averaged over overlap thresholds 0.50 to 0.95, or at 0.50 only.</dd>
<dt>Precision, recall, F1</dt><dd>Of reported detections, the share that are right; of real objects, the share found; and their harmonic mean.</dd>
<dt>Per-channel / per-tensor</dt><dd>One INT8 scale per output channel of each convolution, or one per layer.</dd>
<dt>Calibration</dt><dd>Running sample images through the model to choose the INT8 value ranges. Percentile ignores rare outliers; min-max uses the extremes.</dd>
<dt>Backbone, neck (PANet), head</dt><dd>The three stages of YOLOv5: feature extraction, feature fusion, and the Detect layer that outputs boxes, objects and classes.</dd>
<dt>Compression ratio</dt><dd>FP32 file size divided by the model's file size (same format).</dd>
</dl>

<footer>Generated by <code>scripts/make_report.py</code> from <code>results/*.json</code>. Reproduce with the scripts in <code>scripts/</code>; see <code>docs/METHOD.md</code>.</footer>
</main></body></html>"""
out = R / "report.html"
out.write_text(extra_sections.add_dark_mode(html_doc), encoding="utf-8")
print("wrote", out, f"({out.stat().st_size / 1e3:.0f} KB)")
