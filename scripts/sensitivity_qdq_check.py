"""For each layer in the sensitivity study, count the Quantize/Dequantize nodes inserted when ONLY that layer is
quantized. A layer with 0 was not actually quantized in isolation (e.g. Upsample), so its 'loss' is not meaningful.
    python scripts/sensitivity_qdq_check.py  -> results/sensitivity_qdq.json"""
import json
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import CalibrationMethod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_onnx_variants as B  # noqa: E402

B.install_chunked_calibration()
B.install_calibration_cache(500, CalibrationMethod.Percentile, B.SRC_NORM.stem)
res = {}
for g in range(25):
    others = [f"/model.{i}/" for i in range(25) if i != g]
    p = B.build("m1_all_int8", True, 500, tag=f"qdqcheck_L{g:02d}", exclude_prefixes=others, out_dir=B.WORK / "qdqcheck", src=B.SRC_NORM)
    m = onnx.load(str(p))
    res[f"L{g:02d}"] = sum(n.op_type in ("QuantizeLinear", "DequantizeLinear") and f"/model.{g}/" in (n.name + n.input[0] + n.output[0]) for n in m.graph.node)
    res[f"L{g:02d}_total"] = sum(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in m.graph.node)
    p.unlink()
    print(f"layer {g:2d}: Q/DQ nodes {res[f'L{g:02d}']}", flush=True)
(Path(__file__).resolve().parent.parent / "results" / "sensitivity_qdq.json").write_text(json.dumps(res, indent=1))
