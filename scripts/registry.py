"""One list of every benchmarked model, shared by the latency / memory / evaluation scripts.

key naming:  m0_fp32, m*_int8_(pc|pt)[_norm]  SiLU models, ONNX Runtime
             r0_relu_fp32, r_*                 ReLU models (fine-tuned, head untouched), ONNX Runtime
             tflite_<design>                   TFLite pipelines (TensorFlow venv)
             ov_<onnx key>                     the same ONNX file run with OpenVINO
             ovq_<name>                        OpenVINO IR quantized with NNCF (models/ov_int8)
box_scale: 640 for models whose box output is normalised to 0-1, 1 for pixel-scale boxes.
"""
from pathlib import Path

DAY2 = Path(__file__).resolve().parent.parent
PIXEL_M1 = {"m1_all_int8_pc", "m1_all_int8_pt"}           # collapsed control (pixel boxes); excluded from benchmarks


def jobs(kinds=("onnx", "tflite", "ov")):
    j = [dict(kind="onnx", path=DAY2 / "models/fp32/yolov5n.onnx", key="m0_fp32", box_scale=1.0)]
    for p in sorted((DAY2 / "models/onnx_int8").glob("*.onnx")):
        if not p.stem.startswith("_") and p.stem not in PIXEL_M1:
            j.append(dict(kind="onnx", path=p, key=p.stem, box_scale=640.0 if p.stem.endswith("_norm") else 1.0))
    rf = DAY2 / "models/relu/relu_finetuned_norm.onnx"
    if rf.exists():
        j.append(dict(kind="onnx", path=rf, key="r0_relu_fp32", box_scale=640.0))
    for stem in ("hybrid_s55_norm", "hybrid_s55_v2_norm"):
        hp = DAY2 / "models/relu" / (stem + ".onnx")
        if hp.exists():
            j.append(dict(kind="onnx", path=hp, key="h_" + stem.replace("_norm", "") + "_fp32", box_scale=640.0))
    for p in sorted((DAY2 / "models/relu_int8").glob("*.onnx")) if (DAY2 / "models/relu_int8").exists() else []:
        if not p.stem.startswith("_"):
            j.append(dict(kind="onnx", path=p, key="r_" + p.stem, box_scale=640.0))
    for d in sorted((DAY2 / "models/tflite").glob("*")):
        if (d / "manifest.json").exists():
            j.append(dict(kind="tflite", path=d, key="tflite_" + d.name, box_scale=640.0))
    for p in sorted((DAY2 / "models/ov_int8").glob("*.xml")) if (DAY2 / "models/ov_int8").exists() else []:
        j.append(dict(kind="ov", path=p, key="ovq_" + p.stem, box_scale=640.0))      # OpenVINO IR quantized with NNCF
    onnx_jobs = [x for x in j if x["kind"] == "onnx"]
    j += [dict(x, kind="ov", key="ov_" + x["key"]) for x in onnx_jobs if not x["key"].endswith(("_pt", "_pt_norm"))]   # OpenVINO: per-channel models only
    return [x for x in j if x["kind"] in kinds]
