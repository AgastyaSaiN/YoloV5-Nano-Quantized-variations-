#!/bin/bash
# After a hybrid fine-tune (tag e.g. hybrid_s55): export, FP32 accuracy, NNCF INT8 (M1/M2/M3), INT8 accuracy.
# usage: bash scripts/run_hybrid_pipeline.sh hybrid_s55
cd "$(dirname "$0")/.."
TAG=$1
python scripts/make_relu_model.py --weights models/relu/$TAG.pt --tag $TAG --keep-acts
python scripts/evaluate_onnx.py models/relu/${TAG}_norm.onnx --name ${TAG}_fp32 --box-scale 640 > results/logs/eval_${TAG}_fp32.log 2>&1
python scripts/quantize_openvino.py --src models/relu/${TAG}_norm.onnx --prefix ${TAG}_ --preset mixed > results/logs/quantize_ov_${TAG}.log 2>&1
python scripts/evaluate_ov.py ovq_${TAG}_m1_mixed ovq_${TAG}_m2_mixed ovq_${TAG}_m3_mixed > results/logs/eval_ov_${TAG}.log 2>&1
echo HYBRID_PIPELINE_DONE_$TAG
