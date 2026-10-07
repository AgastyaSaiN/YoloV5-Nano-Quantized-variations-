"""Build the TFLite variants (run with the TensorFlow venv:  .venv\\Scripts\\python.exe scripts/build_tflite_variants.py).

Source: onnx2tf SavedModels in models/tflite/_work/saved/{full,backbone,neck_head,backbone_neck,head}
(made from the normalised-box FP32 ONNX, split at the backbone / neck / head boundaries).

Output: models/tflite/<design>/stageN.tflite + manifest.json  (stages are run in order; INT8 stages take/give float
tensors at their edges, so a pipeline can mix INT8 and FP32 stages).

  m0_fp32                 full float32                               1 stage
  m1_all_int8_{pc,pt}     full model, full-integer INT8              1 stage
  m2_backbone_int8_{pc,pt} backbone INT8  -> neck+head FP32          2 stages
  m3_backbone_neck_int8_{pc,pt} backbone+neck INT8 -> head FP32      2 stages
pc = per-channel weight quantization, pt = per-tensor.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CALIB_DIR, DAY2, preprocess  # noqa: E402

SAVED = DAY2 / "models" / "tflite" / "_work" / "saved"
OUT = DAY2 / "models" / "tflite"
N_CALIB = 500


def representative_dataset(n):
    files = sorted(CALIB_DIR.glob("*.jpg"))[:n]

    def gen():
        for f in files:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield [preprocess(img)[0].transpose(0, 2, 3, 1).astype(np.float32)]   # NHWC
    return gen


def to_int8(saved_dir, per_channel, n_calib=N_CALIB):
    c = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = representative_dataset(n_calib)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]     # strict: any non-INT8 op fails loudly
    if not per_channel:
        c._experimental_disable_per_channel = True
    return c.convert()


def write(design, stages, note):
    d = OUT / design
    d.mkdir(parents=True, exist_ok=True)
    names = []
    for i, blob in enumerate(stages, 1):
        p = d / f"stage{i}.tflite"
        p.write_bytes(blob) if isinstance(blob, bytes) else shutil.copy(blob, p)
        names.append(p.name)
    size = sum((d / n).stat().st_size for n in names) / 1e6
    (d / "manifest.json").write_text(json.dumps({"design": design, "stages": names, "note": note, "size_mb": size}, indent=2))
    print(f"  {design}: {len(names)} stage(s), {size:.2f} MB", flush=True)


if __name__ == "__main__":
    want = sys.argv[1:]
    f32 = lambda part: SAVED / part / f"{part}_float32.tflite"
    jobs = {"m0_fp32": lambda: write("m0_fp32", [f32("full")], "full float32")}
    for g, pc in (("pc", True), ("pt", False)):
        jobs[f"m1_all_int8_{g}"] = (lambda pc=pc, g=g: write(f"m1_all_int8_{g}", [to_int8(SAVED / "full", pc)], "all INT8 incl. head"))
        jobs[f"m2_backbone_int8_{g}"] = (lambda pc=pc, g=g: write(f"m2_backbone_int8_{g}", [to_int8(SAVED / "backbone", pc), f32("neck_head")], "backbone INT8 | neck+head FP32"))
        jobs[f"m3_backbone_neck_int8_{g}"] = (lambda pc=pc, g=g: write(f"m3_backbone_neck_int8_{g}", [to_int8(SAVED / "backbone_neck", pc), f32("head")], "backbone+neck INT8 | head FP32"))
    for name, job in jobs.items():
        if want and name not in want:
            continue
        if (OUT / name / "manifest.json").exists():
            print("skip (exists)", name)
            continue
        print("==", name, flush=True)
        try:
            job()
        except Exception as e:  # keep going so one failure doesn't hide the others
            print("  FAILED:", type(e).__name__, str(e)[:600], flush=True)
