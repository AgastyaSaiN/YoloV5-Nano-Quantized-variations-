# Day 2: INT8 quantization of YOLOv5n for an embedded NPU target

- **[results/benchmark_summary.html](results/benchmark_summary.html)**: the short summary. What was built, then the six benchmark metrics one by one (mAP@0.5, mAP@0.5:0.95, binary size and compression ratio, time per image, precision/recall/F1, memory), each with a chart and a table.
- **[results/report.html](results/report.html)**: the full report with the extra studies (layer sensitivity, calibration, statistics, profile, method).
- **[docs/VERIFICATION.md](docs/VERIFICATION.md)**: the checks made on the results, and the corrections that followed.

## Result in brief

FP32 baseline: **mAP50-95 0.281** on all 5000 COCO val2017 images (matches the published 0.28).

| Design | Backbone | Neck | Head | ONNX per-channel | TFLite per-channel | Size (ONNX / TFLite) |
|---|---|---|---|---|---|---|
| M0 | FP32 | FP32 | FP32 | 0.2811 | 0.2811 | 7.9 MB |
| M1 | INT8 | INT8 | INT8 | 0.2553 | 0.2491 | 2.3 / 2.2 MB |
| M2 | INT8 | FP32 | FP32 | **0.2766** | **0.2728** | 4.9 MB |
| M3 | INT8 | INT8 | FP32 | 0.2735 | 0.2677 | 2.9 / 2.8 MB |

- **Per-channel beats per-tensor** in every pairing (0.8 to 1.3 mAP points, statistically real).
- **The Detect head is the sensitive layer:** quantizing it alone costs 7.0% accuracy; every other single layer that was actually quantized costs 1.0% or less.
- **M1 only works with box output normalised to 0-1.** With pixel-scale boxes it scores 0.000 (confidence scores round to zero).
- **Calibration size barely matters** (50 to 500 images: 0.2721 to 0.2735). **Calibration method does:** percentile calibration scores about 0.6 mAP points above min-max, which accounts for the ONNX-versus-TFLite difference.
- **Speed (desktop CPU, 9 interleaved rounds):** INT8 ONNX models take 31 to 41% less time per image than FP32; all-INT8 TFLite models about 20% less; TFLite designs that keep an FP32 stage 4 to 10% less. The INT8 gain is limited by the unfused SiLU activation and the INT8 neck (see the report).
- **Memory (added to the process during inference):** all-INT8 models use about 17% (ONNX) and 36% (TFLite) less than FP32; designs that mix INT8 and FP32 use 6 to 7% more (ONNX) or 11 to 12% less (TFLite).
- None of the speed or memory figures is an NPU measurement.

## Layout

```
Day 2/
├── README.md
├── requirements.txt
├── docs/                 PLAN.md (original plan), METHOD.md (how everything was built), VERIFICATION.md (checks and corrections)
├── scripts/              all code (see below)
├── vendor/yolov5/        official YOLOv5 v7.0 source
├── models/
│   ├── fp32/             yolov5n.pt / .onnx, yolov5n_norm.onnx (boxes 0-1)
│   ├── onnx_int8/        M1-M3, per-channel (_pc) and per-tensor (_pt); *_norm = M1 with 0-1 boxes
│   └── tflite/           one folder per pipeline: stage1.tflite [, stage2.tflite], manifest.json
├── data/                 500 train2017 calibration images, COCO annotations
├── results/              benchmark_summary.html, report.html, summary.csv/json, accuracy*.json, prf.json, latency.json,
│                         memory.json, profile.json, folds*.json, paired.json, sensitivity*.json, calib_*.json,
│                         charts/, charts_summary/, dets/ (raw detections), logs/
└── .venv/                TensorFlow toolchain (kept separate from the main Python environment)
```

Day 2 reads Day 1's `val2017.zip` in place (about 780 MB) rather than copying it.

## Scripts

| Script | Purpose |
|---|---|
| `prepare_data.py` | 500 seeded-random train2017 images for calibration |
| `make_normalized_model.py` | FP32 ONNX with box output divided by 640 (verified identical to the original) |
| `build_onnx_variants.py` | M1-M3, per-channel and per-tensor, ONNX (`--norm` for the 0-1 box model); calibration cached |
| `split_onnx.py`, `convert_parts_onnx2tf.sh`, `build_tflite_variants.py` | TFLite pipelines (split at backbone / neck / head, convert, quantize) |
| `evaluate_onnx.py`, `evaluate_tflite.py`, `common.py` | COCO accuracy on all 5000 images |
| `compute_prf.py` | precision / recall / F1 with the official COCO matching |
| `benchmark_latency.py` | time per image, interleaved rounds |
| `benchmark_memory.py`, `aggregate_memory.py` | memory added to the process, repeated and aggregated |
| `profile_ort.py` | per-region, per-operation time in ONNX Runtime |
| `check_head_untouched.py` | confirms the Detect head weights are identical to the original in every FP32-head design |
| `fold_stats.py`, `paired_comparisons.py` | 10-fold confidence intervals and paired comparisons |
| `layer_sensitivity.py`, `sensitivity_qdq_check.py` | per-layer sensitivity, and which layers were truly quantized |
| `calib_size_study.py`, `calib_method_study.py` | calibration-set size and calibration method |
| `make_summary.py`, `make_charts.py`, `make_report.py`, `make_benchmark_summary.py` | summary table, charts, full report, one-page summary |
| `run_*.sh` | convenience chains |

## Limitations

- **No NPU or board was used.** Timing and memory are desktop-CPU figures. On a real NPU the FP32 stages of M2 and M3 fall back to the host CPU.
- Latency is noisy (the same model varies by up to about 27% between rounds); gaps between INT8 variants are not meaningful.
- Layer sensitivity uses the first 1000 val images and one layer at a time; upsample layers could not be quantized in isolation.
- Quantization-aware training was not attempted.
