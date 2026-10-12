#!/bin/bash
# Model fine-tuned at a reduced input size (share of activations replaced by ReLU from $1; 0 keeps SiLU), then exported at that size, scored in FP32 and INT8 (OpenVINO NNCF).
# Resumable: run again after an interruption.   usage: bash scripts/run_r512_chain.sh SHARE TAG [SIZE=512] [EPOCHS=24]   e.g. 0 ft_448 448 20
cd "$(dirname "$0")/.."
SHARE=${1:-0.55}; TAG=${2:-r55_512}; SIZE=${3:-512}; EPOCHS=${4:-24}
export YOLO_IMGSZ=$SIZE
.venv_train/Scripts/python.exe scripts/finetune_relu.py --imgsz $SIZE --epochs $EPOCHS --batch 16 --workers 4 --anneal-frac 0.4 --init silu --replace-share $SHARE --gt-weight 1.0 --ema --resume --out models/relu/$TAG.pt >> results/logs/finetune_$TAG.log 2>&1
python scripts/make_relu_model.py --weights models/relu/$TAG.pt --tag $TAG --keep-acts --imgsz $SIZE
python scripts/evaluate_onnx.py models/relu/${TAG}_norm.onnx --name ${TAG}_fp32 --box-scale $SIZE > results/logs/eval_${TAG}_fp32.log 2>&1
python scripts/quantize_openvino.py --src models/relu/${TAG}_norm.onnx --prefix ${TAG}_ --preset mixed > results/logs/quantize_ov_${TAG}.log 2>&1
python scripts/evaluate_ov.py ovq_${TAG}_m1_mixed ovq_${TAG}_m3_mixed > results/logs/eval_ov_${TAG}.log 2>&1
echo done > results/logs/$TAG.done
