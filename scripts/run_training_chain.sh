#!/bin/bash
# Waits for the training-image download to finish, then fine-tunes the ReLU model (GPU environment).
cd "$(dirname "$0")/.."
until [ -f data/train2017/list.txt ]; do sleep 20; done
echo "download complete: $(ls data/train2017/images | wc -l) images"
.venv_train/Scripts/python.exe scripts/finetune_relu.py --epochs 15 --batch 16 --workers 6
echo TRAINING_DONE
