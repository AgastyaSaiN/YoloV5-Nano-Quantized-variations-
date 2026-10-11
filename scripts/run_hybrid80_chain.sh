#!/bin/bash
# Hybrid with a larger share of ReLU activations (80% of the SiLU work), fine-tuned like hybrid_s55_v2, then exported and scored.
# Resumable: run again after an interruption.
cd "$(dirname "$0")/.."
.venv_train/Scripts/python.exe scripts/finetune_relu.py --epochs 30 --batch 16 --workers 6 --anneal-frac 0.4 --init silu --replace-share 0.80 --gt-weight 1.0 --ema --resume --out models/relu/hybrid_s80.pt >> results/logs/finetune_hybrid_s80.log 2>&1
bash scripts/run_hybrid_pipeline.sh hybrid_s80
