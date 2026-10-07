#!/bin/bash
# Build all TFLite variants, then score each on all 5000 val images (TensorFlow venv).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
$PY scripts/build_tflite_variants.py
for d in models/tflite/m*/; do
  n=tflite_$(basename "$d")
  $PY -c "import json,sys;sys.exit(0 if '$n' in json.load(open('results/accuracy_tflite.json')) else 1)" 2>/dev/null && { echo "skip $n"; continue; }
  echo "== eval $n"; $PY scripts/evaluate_tflite.py "$d" > "results/logs/eval_$n.log" 2>&1
done
echo ALL_TFLITE_DONE
