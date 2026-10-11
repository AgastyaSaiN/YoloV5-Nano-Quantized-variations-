"""The results told stage by stage: what was done, why, and what it achieved. Shared by report.html and benchmark_summary.html.
Reads results/*.json (through extra_sections). Use story_html(img_tag), story_chart(out_dir) and stage_note(metric)."""
import html
from pathlib import Path

import extra_sections as E
from extra_sections import BASE, BL, L, acc, mem, table

prf = E.J("prf.json")
ACC_KEY = E.ACC_KEY


def _prf(key, name):
    p = prf[key]
    return p.get(name + "@0.25", p.get({"P": "precision", "R": "recall", "F1": "F1"}[name] + "@0.25"))


STAGES = [
    dict(n=1, title="First INT8 models", key="m3_backbone_neck_int8_pc", base="m0_fp32", rt="ONNX Runtime", short="M3 per-channel",
         did="Built the three designs (M1 everything INT8, M2 backbone INT8, M3 backbone and neck INT8; Detect head FP32 in M2 and M3), each per-channel and per-tensor, in ONNX and TFLite. Scored on all 5000 COCO val2017 images.",
         got="Accuracy held up well (M2 98%, M3 97% of FP32 in ONNX), files shrank 2.7 to 3.4x, but the speed-up was small. ONNX and TFLite gave different accuracy for the same design."),
    dict(n=2, title="Audit and corrections", key="m3_backbone_neck_int8_pc", base="m0_fp32", rt="ONNX Runtime", short="M3 per-channel, re-measured",
         did="Questioned every number. Found and fixed: unpinned threads that inflated the FP32 baseline by about 50%, a timing run taken while training was running, a calibration cache that froze value ranges, and a precision/recall matcher that ignored crowd regions. Re-measured everything on an idle machine with four pinned cores.",
         got="No model changed, but the numbers became trustworthy. The ONNX-versus-TFLite gap turned out to be the calibration method (percentile versus min-max): with the method matched, the formats agree within about 0.2 mAP points."),
    dict(n=3, title="OpenVINO quantization", key="ovq_m3_mixed", base="ov_m0_fp32", rt="OpenVINO", short="M3, NNCF",
         did="Quantized the same network with OpenVINO and its NNCF tool, which handles activations better than the ONNX Runtime QDQ format.",
         got="Faster than ONNX Runtime and the smallest files, with 97.7% of the accuracy. Still limited by the SiLU activation, which stays a separate sigmoid and multiply in INT8."),
    dict(n=4, title="ReLU in the largest layers (R55)", key="ovq_hybrid_s55_v2_m3_mixed", base="ov_m0_fp32", rt="OpenVINO", short="R55 M3, NNCF",
         did="Replaced the 12 largest SiLU activations (about 55% of the activation work: layers 0 to 4 and 17) with ReLU and fine-tuned the network with the Detect head frozen and unchanged (verified parameter by parameter).",
         got="ReLU fuses into the INT8 convolution, so the INT8 speed-up rose clearly. About 5 points of accuracy were given up for it."),
    dict(n=5, title="More ReLU (R80)", key="ovq_hybrid_s80_m3_mixed", base="ov_m0_fp32", rt="OpenVINO", short="R80 M3, NNCF",
         did="Replaced 29 of the 57 activations (81% of the activation work) and fine-tuned for 30 epochs.",
         got="Faster again, and the all-INT8 R80 model (M1) passes 2x; but each extra step costs about 3 to 4 accuracy points, so this is the fast end of the trade-off."),
]


def _metrics(key, base):
    a = acc[ACC_KEY.get(key, key)]
    k = ACC_KEY.get(key, key)
    return dict(map50=a["mAP50"], map=a["mAP50-95"], ret=a["mAP50-95"] / BASE * 100, size=a["size_mb"], comp=acc["m0_fp32"]["size_mb"] / a["size_mb"],
                ms=L(key), spd=L(base) / L(key), p=_prf(k, "P"), r=_prf(k, "R"), f1=_prf(k, "F1"), mem=mem.get(key, {}).get("peak_mb"))


for _s in STAGES:
    _s["m"] = _metrics(_s["key"], _s["base"])
BASE_M = _metrics("m0_fp32", "m0_fp32")


def story_chart(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = list(range(len(STAGES)))
    vals = [1.5] + [s["m"]["spd"] for s in STAGES[1:]]
    bars = ax.bar(x, vals, color="#1f4e9e", width=0.55)
    bars[0].set_hatch("//"); bars[0].set_facecolor("#9db5dc")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}x" + ("*" if i == 0 else ""), ha="center", fontsize=9)
    ax.text(0.0, -0.30, "* first measurement (unpinned threads), later found to overstate the speed-up", transform=ax.transAxes, fontsize=8, color="#666")
    ax.set_ylabel("speed-up over FP32"); ax.set_ylim(0, 2.4); ax.axhline(1, color="#999", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"Stage {s['n']}\n" + s["title"].replace(" (", "\n(").replace("Audit and ", "Audit and\n") for s in STAGES], fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(x, [s["m"]["ret"] for s in STAGES], color="#c0551b", marker="o"); ax2.set_ylim(80, 103)
    for i, s in enumerate(STAGES):
        ax2.text(i, s["m"]["ret"] + 0.9, f"{s['m']['ret']:.1f}%", ha="center", color="#c0551b", fontsize=9)
    ax2.set_ylabel("accuracy retained (%)", color="#c0551b")
    ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "9_progress_by_stage.png", dpi=150); plt.close(fig)
    return "9_progress_by_stage.png"


def story_html(img_tag):
    fmt = lambda m: f"{m['p']:.3f} / {m['r']:.3f} / {m['f1']:.3f}"
    rows = [["0", "FP32 baseline (ONNX Runtime)", f"{BASE_M['map50']:.3f}", f"{BASE_M['map']:.4f}", "100%", f"{BASE_M['size']:.2f} MB / 1.0x", f"{BASE_M['ms']:.1f}", "1.00x", fmt(BASE_M), f"{BASE_M['mem']:.0f}"]]
    for s in STAGES:
        m = s["m"]
        ms, spd = (f"{m['ms']:.1f}", f"{m['spd']:.2f}x") if s["n"] != 1 else ("first run: 36.5 FP32, unpinned", "about 1.5x, overstated")
        rows.append([str(s["n"]), f"{s['title']}: {s['short']} ({s['rt']})", f"{m['map50']:.3f}", f"{m['map']:.4f}", f"{m['ret']:.1f}%", f"{m['size']:.2f} MB / {m['comp']:.1f}x",
                     ms, spd, fmt(m), "" if m["mem"] is None else f"{m['mem']:.0f}"])
    tab = table(["Stage", "Model shown", "mAP50", "mAP50-95", "Retained", "Size / compression", "ms per image", "Speed-up", "Precision / recall / F1", "Peak memory added (MB)"], rows,
                "One representative model per stage, all measured with the same protocol. Speed-up is against the FP32 model of the same runtime; precision, recall and F1 at confidence 0.25.")
    steps = "".join(f"<h3>Stage {s['n']}: {html.escape(s['title'])}</h3><p><b>What we did.</b> {s['did']}</p><p><b>What came out.</b> {s['got']} "
                    f"For {html.escape(s['short'])}: {s['m']['ret']:.1f}% of the FP32 accuracy, {s['m']['spd']:.2f}x faster, {s['m']['size']:.2f} MB.</p>" for s in STAGES)
    s3, s4, s5 = STAGES[2]["m"], STAGES[3]["m"], STAGES[4]["m"]
    return f"""<h2>The story so far: five stages</h2>
<p>The work happened in stages, each one prompted by what the previous one showed. The table follows one representative design (M3: backbone and neck INT8, Detect head FP32) through the stages. The Detect head is untouched at every stage.</p>
<figure>{img_tag}<figcaption><b>Progress by stage.</b> Bars: speed-up over the FP32 model of the same runtime. Line: share of the FP32 accuracy kept.</figcaption></figure>
{tab}
{steps}
<h3>Where this leaves us</h3>
<p>Quantization alone gave a modest speed-up on this CPU (stages 1 to 3, up to {s3['spd']:.2f}x) while shrinking the file about {s3['comp']:.0f}x. Changing the largest activations to ReLU (stages 4 and 5) raised the speed to {s4['spd']:.2f}x and {s5['spd']:.2f}x and, in the all-INT8 R80 model, past 2x, at the price of accuracy. The remaining time is spent in the convolutions, in the neck's resize and concatenate steps, and in pre- and post-processing, which quantization does not help. The next steps are channel pruning, quantization-aware training and faster post-processing.</p>"""


def stage_note(metric):
    """One sentence per benchmark: how it changed from stage 1 to stage 5."""
    a, b, c = STAGES[0]["m"], STAGES[2]["m"], STAGES[4]["m"]
    return {
        "map50": f"<b>Across the stages:</b> mAP@0.5 went from {a['map50']:.3f} (stage 1) to {b['map50']:.3f} (OpenVINO, stage 3) to {c['map50']:.3f} (R80, stage 5), against {BASE_M['map50']:.3f} for FP32.",
        "map": f"<b>Across the stages:</b> {a['ret']:.1f}% of the FP32 mAP@0.5:0.95 was kept at stage 1, {b['ret']:.1f}% at stage 3 and {c['ret']:.1f}% at stage 5; the ReLU stages trade accuracy for speed.",
        "size": f"<b>Across the stages:</b> the file went from {BASE_M['size']:.1f} MB (FP32) to {a['size']:.2f} MB (stage 1) and {b['size']:.2f} MB (OpenVINO, stage 3, {b['comp']:.1f}x compression); ReLU does not change the size.",
        "time": f"<b>Across the stages:</b> speed-up over FP32 was {a['spd']:.2f}x once measured properly (stage 2; the first, unpinned measurement had overstated it), {b['spd']:.2f}x with OpenVINO (stage 3), {STAGES[3]['m']['spd']:.2f}x with R55 and {c['spd']:.2f}x with R80 (the all-INT8 R80 model reaches 2x).",
        "prf": f"<b>Across the stages:</b> F1 at confidence 0.25 went from {BASE_M['f1']:.3f} (FP32) to {a['f1']:.3f} (stage 1), {b['f1']:.3f} (stage 3) and {c['f1']:.3f} (stage 5); recall drops more than precision.",
        "mem": f"<b>Across the stages:</b> peak memory barely changed with INT8 on this CPU (activations dominate it); OpenVINO uses more than ONNX Runtime in absolute terms because of its own runtime buffers.",
    }[metric]
