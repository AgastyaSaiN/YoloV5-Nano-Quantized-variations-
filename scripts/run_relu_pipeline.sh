#!/bin/bash
# After fine-tuning: export the ReLU model, build every INT8 variant (ONNX + TFLite), and score them all.
cd "$(dirname "$0")/.."
SRC=models/relu/relu_finetuned_norm.onnx
echo "== export"; python scripts/make_relu_model.py --weights models/relu/relu_trained.pt --tag relu_finetuned
echo "== FP32 ReLU accuracy"; python scripts/evaluate_onnx.py $SRC --name r0_relu_fp32 --box-scale 640 > results/logs/eval_r0_relu_fp32.log 2>&1
echo "== ONNX INT8 builds"; python scripts/build_onnx_variants.py --src $SRC --out-dir models/relu_int8 > results/logs/build_relu_int8.log 2>&1
for m in models/relu_int8/m*.onnx; do n=r_$(basename "$m" .onnx); echo "== eval $n"; python scripts/evaluate_onnx.py "$m" --name "$n" --box-scale 640 > "results/logs/eval_$n.log" 2>&1; done
echo "== TFLite"
python scripts/split_onnx.py --src $SRC --tag relu > results/logs/split_relu.log 2>&1
bash scripts/convert_parts_onnx2tf.sh relu > results/logs/convert_relu.log 2>&1
.venv/Scripts/python.exe scripts/build_tflite_variants.py --relu > results/logs/build_tflite_relu.log 2>&1
for d in models/tflite/r_m*/; do echo "== eval tflite $(basename $d)"; .venv/Scripts/python.exe scripts/evaluate_tflite.py "$d" > "results/logs/eval_tflite_$(basename $d).log" 2>&1; done
echo RELU_PIPELINE_DONE
