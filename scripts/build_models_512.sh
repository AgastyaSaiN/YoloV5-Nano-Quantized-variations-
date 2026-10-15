#!/bin/bash
# Build and score every INT8 model at 512 x 512 from the trained YOLOv5n (models/fp32/yolov5n.pt). Resumable: finished models are skipped.
# Needs the main Python (ONNX Runtime, OpenVINO + NNCF, LiteRT, PyTorch) and the TensorFlow environment (.venv) for the TFLite conversion.
# Afterwards:  bash scripts/run_all_benchmarks.sh   (latency, memory, post-processing, tables, charts, reports)
cd "$(dirname "$0")/.."
mkdir -p results/logs
export YOLO_IMGSZ=512
PY_TF=.venv/Scripts/python.exe

# 1. FP32 export at 512 px (boxes normalised to 0-1) and its accuracy
[ -f models/fp32_512/f512_norm.onnx ] || python scripts/export_onnx.py --imgsz 512 --tag f512
python -c "import json,sys;sys.exit(0 if 'f512_fp32' in json.load(open('results/accuracy.json')) else 1)" 2>/dev/null || \
  python scripts/evaluate_onnx.py models/fp32_512/f512_norm.onnx --name f512_fp32 --box-scale 512 > results/logs/eval_f512_fp32.log 2>&1

# 2. ONNX Runtime INT8 (M1/M2/M3 x per-channel/per-tensor, QDQ, percentile calibration)
python scripts/build_onnx_variants.py --src models/fp32_512/f512_norm.onnx --out-dir models/onnx_int8_512 > results/logs/build_onnx_512.log 2>&1
for m in models/onnx_int8_512/m*.onnx; do
  n=s512_$(basename "$m" .onnx)
  python -c "import json,sys;sys.exit(0 if '$n' in json.load(open('results/accuracy.json')) else 1)" 2>/dev/null && continue
  python scripts/evaluate_onnx.py "$m" --name $n --box-scale 512 > results/logs/eval_$n.log 2>&1
done

# 3. OpenVINO INT8 (NNCF post-training quantization)
python scripts/quantize_openvino.py --src models/fp32_512/f512_norm.onnx --prefix f512_ --preset mixed > results/logs/quantize_ov_512.log 2>&1
python scripts/evaluate_ov.py ovq_f512_m1_mixed ovq_f512_m2_mixed ovq_f512_m3_mixed > results/logs/eval_ov_512.log 2>&1

# 4. TFLite INT8: split the graph (used to locate the float layers), convert with onnx2tf, build one .tflite file per design, score
$PY_TF scripts/split_onnx.py --src models/fp32_512/f512_norm.onnx --tag 512 > results/logs/split_512.log 2>&1
bash scripts/convert_parts_onnx2tf.sh 512 > results/logs/convert_512.log 2>&1
$PY_TF scripts/build_tflite_single.py --size 512 > results/logs/build_tflite_512.log 2>&1
$PY_TF scripts/build_tflite_single.py --size 512 --int-io > results/logs/build_tflite_io8.log 2>&1      # M1 with integer input/output
# the FP32 TFLite models (640 px reference and 512 px) come straight from the converter
[ -f models/tflite/s512_m0_fp32/manifest.json ] || python - <<'PY'
import json, shutil
from pathlib import Path
d = Path("models/tflite/s512_m0_fp32"); d.mkdir(parents=True, exist_ok=True)
shutil.copy("models/tflite/_work/512/saved/full/full_float32.tflite", d / "stage1.tflite")
(d / "manifest.json").write_text(json.dumps({"design": "s512_m0_fp32", "stages": ["stage1.tflite"], "note": "full float32", "size_mb": (d / "stage1.tflite").stat().st_size / 1e6}, indent=2))
PY
for d in models/tflite/t512_m*/; do
  n=tflite_$(basename "$d")
  python -c "import json,sys;sys.exit(0 if '$n' in json.load(open('results/accuracy_tflite.json')) else 1)" 2>/dev/null && continue
  python scripts/evaluate_tflite.py "$d" > "results/logs/eval_$n.log" 2>&1
done

# 5. precision / recall / F1 from the saved detections
python scripts/compute_prf.py > results/logs/prf.log 2>&1
echo "models built and scored"
