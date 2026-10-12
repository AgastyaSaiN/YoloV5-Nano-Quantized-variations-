#!/bin/bash
# Plain YOLOv5n (SiLU), INT8 with OpenVINO NNCF at reduced input sizes: export, quantize (500 calibration images at that size), score on all 5000 val images.
cd "$(dirname "$0")/.."
for N in "$@"; do
  export YOLO_IMGSZ=$N
  python scripts/make_relu_model.py --weights models/fp32/yolov5n.pt --tag f$N --keep-acts --imgsz $N
  python scripts/quantize_openvino.py --src models/relu/f${N}_norm.onnx --prefix f${N}_ --preset mixed > results/logs/quantize_ov_f$N.log 2>&1
  python scripts/evaluate_ov.py ovq_f${N}_m1_mixed ovq_f${N}_m3_mixed > results/logs/eval_ov_f$N.log 2>&1
done
echo done > results/logs/resolution.done
