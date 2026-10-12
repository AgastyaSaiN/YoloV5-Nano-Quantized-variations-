#!/bin/bash
# Re-score the two TFLite M2 pipelines at 512 px after the layout hint was added, then run the final update.
cd "$(dirname "$0")/.."
export YOLO_IMGSZ=512
for g in pc pt; do .venv/Scripts/python.exe scripts/evaluate_tflite.py models/tflite/s512_m2_backbone_int8_$g > results/logs/eval_tflite_s512_m2_$g.log 2>&1; done
unset YOLO_IMGSZ
bash scripts/run_final_update.sh
