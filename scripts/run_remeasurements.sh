#!/bin/bash
# Re-measurements (P/R/F1, quantized-layer check, memory x3, latency rounds). Run on an otherwise idle machine, in this order.
cd "$(dirname "$0")/.."
python scripts/compute_prf.py
python scripts/sensitivity_qdq_check.py
rm -f results/memory.json
for i in 1 2 3; do python scripts/benchmark_memory.py > results/logs/memory_run$i.log 2>&1; cp results/memory.json results/logs/memory_run$i.json; rm results/memory.json; done
python scripts/benchmark_latency.py --rounds 9
echo REMEASUREMENTS_DONE
