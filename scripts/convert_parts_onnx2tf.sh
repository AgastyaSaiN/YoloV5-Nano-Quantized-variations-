#!/bin/bash
# ONNX parts -> TF SavedModels + float32 TFLite with onnx2tf (TensorFlow venv). Run after split_onnx.py.
# usage: bash scripts/convert_parts_onnx2tf.sh [tag]      (tag = relu for the ReLU model; empty for the SiLU model)
# onnx2tf looks for a sample file in the working directory before trying to download one, so a local placeholder is created.
TAG="$1"
cd "$(dirname "$0")/../models/tflite/_work" || exit 1
[ -n "$TAG" ] && cd "$TAG"
export PATH="$(cd "$(dirname "$0")/../../../../.venv/Scripts" 2>/dev/null && pwd):$PATH"
export PATH="/c/Users/soory/Downloads/Yolo thing/Day 2/.venv/Scripts:$PATH"
python -c "
import numpy as np
np.save('calibration_image_sample_data_20x128x128x3_float32.npy', np.random.RandomState(0).rand(20,128,128,3).astype(np.float32))"
mkdir -p saved
for p in full backbone neck_head backbone_neck head; do
  rm -rf saved/$p
  onnx2tf.exe -i onnx_parts/$p.onnx -o saved/$p -osd > "/c/Users/soory/Downloads/Yolo thing/Day 2/results/logs/onnx2tf_${TAG}_$p.log" 2>&1
  echo "$p exit=$?"
done
