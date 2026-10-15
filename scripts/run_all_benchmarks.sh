#!/bin/bash
# Re-measure speed and memory of EVERY model in one session (idle machine, AC power), then rebuild the tables, charts and both reports.
# Order: latency (9 interleaved rounds), memory x3, TFLite memory probe, post-processing, summary, charts, reports.
cd "$(dirname "$0")/.."
mkdir -p results/logs
cp results/latency.json results/logs/latency_before_rerun.json
python scripts/benchmark_latency.py --rounds 9 > results/logs/latency_rerun.log 2>&1
rm -f results/memory.json
for i in 1 2 3; do python scripts/benchmark_memory.py > results/logs/memory_run$i.log 2>&1; cp results/memory.json results/logs/memory_run$i.json; rm results/memory.json; done
python scripts/aggregate_memory.py > /dev/null
python scripts/probe_tflite_memory.py > results/logs/tflite_memory_probe.log 2>&1
python scripts/benchmark_postprocess.py > results/logs/postprocess.log 2>&1
python scripts/make_summary.py > results/logs/make_summary.log 2>&1
python scripts/make_charts.py > results/logs/make_charts.log 2>&1
python scripts/make_report.py > results/logs/make_report.log 2>&1
python scripts/make_benchmark_summary.py > results/logs/make_benchmark_summary.log 2>&1
echo done > results/logs/full_rerun.done
