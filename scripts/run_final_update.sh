#!/bin/bash
# After the current-stage ONNX/TFLite models are scored: speed (ONNX Runtime + TFLite, interleaved with their 640 px FP32 baselines), memory x3,
# precision/recall, then rebuild the summary table, charts and both reports.
cd "$(dirname "$0")/.."
cp results/latency.json results/logs/latency_before_current_stage.json
python scripts/benchmark_latency.py --rounds 9 --kinds onnx,tflite > results/logs/latency_current_stage.log 2>&1
python - <<'PY'
import json
old = json.load(open('results/logs/latency_before_current_stage.json')); new = json.load(open('results/latency.json'))
m = {k: v for k, v in old.items() if k.startswith('ov')}; m.update(new)
json.dump(m, open('results/latency.json', 'w'), indent=2)
PY
rm -f results/memory.json
for i in 1 2 3; do python scripts/benchmark_memory.py > results/logs/memory_run$i.log 2>&1; cp results/memory.json results/logs/memory_run$i.json; rm results/memory.json; done
python scripts/aggregate_memory.py > /dev/null
python scripts/compute_prf.py > results/logs/prf.log 2>&1
python scripts/make_summary.py > results/logs/make_summary.log 2>&1
python scripts/make_charts.py > results/logs/make_charts.log 2>&1
python scripts/make_report.py > results/logs/make_report.log 2>&1
python scripts/make_benchmark_summary.py > results/logs/make_benchmark_summary.log 2>&1
echo done > results/logs/final_update.done
