#!/bin/bash
# After fine-tuning: export the ReLU model and score it in FP32 (decides whether the INT8 stage is worth running).
cd "$(dirname "$0")/.."
python scripts/make_relu_model.py --weights models/relu/relu_trained.pt --tag relu_finetuned
python scripts/evaluate_onnx.py models/relu/relu_finetuned_norm.onnx --name r0_relu_fp32 --box-scale 640 > results/logs/eval_r0_relu_fp32.log 2>&1
echo STAGE1_DONE
