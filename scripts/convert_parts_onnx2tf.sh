#!/bin/bash
# ONNX parts -> TF SavedModels + float32 TFLite with onnx2tf (TensorFlow venv). Run after split_onnx.py.
# onnx2tf looks for a sample file in the working directory before trying to download one, so a local placeholder is created.
cd "$(dirname "$0")/../models/tflite/_work" || exit 1
export PATH="$(cd ../../../.venv/Scripts && pwd):$PATH"
python -c "
import numpy as np
np.save('calibration_image_sample_data_20x128x128x3_float32.npy', np.random.RandomState(0).rand(20,128,128,3).astype(np.float32))"
cp ../../fp32/yolov5n_norm.onnx onnx_parts/full.onnx
for p in full backbone neck_head backbone_neck head; do
  rm -rf saved/$p
  onnx2tf.exe -i onnx_parts/$p.onnx -o saved/$p -osd > ../../../results/logs/onnx2tf_$p.log 2>&1
  echo "$p exit=$?"
done
