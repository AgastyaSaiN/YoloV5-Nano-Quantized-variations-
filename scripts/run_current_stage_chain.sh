#!/bin/bash
# Current stage for the ONNX Runtime and TFLite models: the same designs (M1/M2/M3, per-channel and per-tensor) built at 512 x 512 input
# from the normalised-box FP32 export (models/relu/f512_norm.onnx), scored on all 5000 val images. Resumable (skips finished models).
cd "$(dirname "$0")/.."
export YOLO_IMGSZ=512
# ONNX Runtime
python scripts/build_onnx_variants.py --src models/relu/f512_norm.onnx --out-dir models/onnx_int8_512 > results/logs/build_onnx_512.log 2>&1
for m in models/onnx_int8_512/m*.onnx; do
  n=s512_$(basename "$m" .onnx)
  python -c "import json,sys;sys.exit(0 if '$n' in json.load(open('results/accuracy.json')) else 1)" 2>/dev/null && continue
  python scripts/evaluate_onnx.py "$m" --name $n --box-scale 512 > results/logs/eval_$n.log 2>&1
done
# TFLite
PY=.venv/Scripts/python.exe
$PY scripts/split_onnx.py --src models/relu/f512_norm.onnx --tag 512 > results/logs/split_512.log 2>&1
bash scripts/convert_parts_onnx2tf.sh 512 > results/logs/convert_512.log 2>&1
$PY scripts/build_tflite_variants.py --tag 512 > results/logs/build_tflite_512.log 2>&1
for d in models/tflite/s512_m*/; do
  n=tflite_$(basename "$d")
  $PY -c "import json,sys;sys.exit(0 if '$n' in json.load(open('results/accuracy_tflite.json')) else 1)" 2>/dev/null && continue
  $PY scripts/evaluate_tflite.py "$d" > "results/logs/eval_$n.log" 2>&1
done
echo done > results/logs/current_stage.done
