"""Build results/report.html: a self-contained, plain white report (charts embedded as images)."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results" / "benchmark_results.json").read_text())
base = R["fp32_reference"]

LABEL = {"fp32_reference": "Original", "v1_percentile_head": "v1", "v2_minmax_head": "v2",
         "v3_percentile9999_head": "v3", "v4_percentile_head_neck": "v4", "ctrl_percentile_all": "Control"}
ORDER = ["fp32_reference", "v1_percentile_head", "v2_minmax_head", "v3_percentile9999_head",
         "v4_percentile_head_neck", "ctrl_percentile_all"]


def img(name):
    data = (ROOT / "results" / "charts_for_report" / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


rows = []
for k in ORDER:
    d = R[k]
    loss = (base["mAP50-95"] - d["mAP50-95"]) / base["mAP50-95"] * 100
    cls = ' class="rec"' if k == "v1_percentile_head" else ""
    rows.append(
        f'    <tr{cls}><td>{LABEL[k]}</td><td>{d["mAP50-95"]:.4f}</td><td>{d["mAP50"]:.4f}</td>'
        f'<td>{d["precision"]:.3f}</td><td>{d["recall"]:.3f}</td>'
        f'<td>{"-" if k == "fp32_reference" else f"{loss:.1f}%"}</td>'
        f'<td>{d["latency_ms_median"]:.1f}</td><td>{d["size_mb"]:.1f}</td></tr>')

v1 = R["v1_percentile_head"]
ticks = "".join(f'<line x1="{20 + i * 720 / 64:.1f}" y1="52" x2="{20 + i * 720 / 64:.1f}" y2="68"/>' for i in range(1, 64))
fill = {
    "{{SIZE_CUT_V1}}": f'{100 - v1["size_mb"] / base["size_mb"] * 100:.0f}',
    "{{FASTER_V1}}": f'{100 - v1["latency_ms_median"] / base["latency_ms_median"] * 100:.0f}',
    "{{LOSS_V1}}": f'{(base["mAP50-95"] - v1["mAP50-95"]) / base["mAP50-95"] * 100:.1f}',
    "{{MAP_V1}}": f'{v1["mAP50-95"]:.3f}',
    "{{MAP_BASE}}": f'{base["mAP50-95"]:.3f}',
    "{{TABLE_ROWS}}": "\n".join(rows),
    "{{TICKS}}": ticks,
    "{{IMG1}}": img("1_accuracy.png"),
    "{{IMG2}}": img("2_size_and_speed.png"),
    "{{IMG3}}": img("3_accuracy_lost_zoomed.png"),
}
html = (ROOT / "scripts" / "report_template.html").read_text(encoding="utf-8")
for k, v in fill.items():
    html = html.replace(k, v)
out = ROOT / "results" / "report.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"({out.stat().st_size / 1e3:.0f} KB)")
