"""Write docs/PROGRESS.md: the results stage by stage (what was done, why, what came out). Reads results/*.json via story.py."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story  # noqa: E402

D = Path(__file__).resolve().parent.parent
fmt = lambda m: f"{m['p']:.3f} / {m['r']:.3f} / {m['f1']:.3f}"
b = story.BASE_M
lines = ["# Progress, stage by stage", "",
         "Each stage was prompted by what the previous one showed. One representative design (M3: backbone and neck INT8, Detect head FP32) is followed through the stages; the Detect head is untouched at every stage. All numbers are on all 5000 COCO val2017 images.", "",
         "| Stage | Model shown | mAP50 | mAP50-95 | Retained | Size / compression | ms per image | Speed-up | P / R / F1 @0.25 | Peak memory added (MB) |", "|---|---|---|---|---|---|---|---|---|---|",
         f"| 0 | FP32 baseline (ONNX Runtime) | {b['map50']:.3f} | {b['map']:.4f} | 100% | {b['size']:.2f} MB / 1.0x | {b['ms']:.1f} | 1.00x | {fmt(b)} | {b['mem']:.0f} |"]
for s in story.STAGES:
    m = s["m"]
    ms, sp = (f"{m['ms']:.1f}", f"{m['spd']:.2f}x") if s["n"] != 1 else ("first run, unpinned (overstated)", "about 1.5x (overstated)")
    lines.append(f"| {s['n']} | {s['title']}: {s['short']} ({s['rt']}) | {m['map50']:.3f} | {m['map']:.4f} | {m['ret']:.1f}% | {m['size']:.2f} MB / {m['comp']:.1f}x | {ms} | {sp} | {fmt(m)} | {'' if m['mem'] is None else round(m['mem'])} |")
lines += [""]
for s in story.STAGES:
    clean = lambda t: re.sub(r"<[^>]+>", "", t)
    lines += [f"## Stage {s['n']}: {s['title']}", "", f"**What we did.** {clean(s['did'])}", "", f"**What came out.** {clean(s['got'])}", ""]
(D / "docs" / "PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote docs/PROGRESS.md")
