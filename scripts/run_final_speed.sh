#!/bin/bash
# Final speed + memory measurements on an idle machine: latency (9 interleaved rounds), then memory x3.
cd "$(dirname "$0")/.."
python scripts/benchmark_latency.py --rounds 9 > results/logs/latency_final.log 2>&1
rm -f results/memory.json
for i in 1 2 3; do python scripts/benchmark_memory.py > results/logs/memory_run$i.log 2>&1; cp results/memory.json results/logs/memory_run$i.json; rm results/memory.json; done
echo FINAL_SPEED_DONE > results/logs/final_speed.done
