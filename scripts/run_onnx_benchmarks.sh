#!/bin/bash
# Accuracy on all 5000 val images for every INT8 ONNX model. (Speed and memory: benchmark_latency.py, benchmark_memory.py.)
cd "$(dirname "$0")/.."
for m in models/onnx_int8/m*.onnx; do
  n=$(basename "$m" .onnx)
  python -c "import json,sys;sys.exit(0 if '$n' in json.load(open('results/accuracy.json')) else 1)" 2>/dev/null && { echo "skip $n"; continue; }
  echo "== eval $n"; python scripts/evaluate_onnx.py "$m" > "results/logs/eval_$n.log" 2>&1
done
echo ALL_ONNX_DONE
