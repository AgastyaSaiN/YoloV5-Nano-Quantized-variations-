# YOLOv5n INT8 quantization benchmark

The original YOLOv5n (nano) and three INT8 versions of it, benchmarked in three runtimes (ONNX Runtime, TFLite, OpenVINO) on a laptop CPU, scored on all 5000 COCO val2017 images. The Detect head (the last layer) is never changed.

| Design | Backbone | Neck | Detect head |
|---|---|---|---|
| FP32 | FP32 | FP32 | FP32 |
| M1 | INT8 | INT8 | INT8 |
| M2 | INT8 | FP32 | FP32 |
| M3 | INT8 | INT8 | FP32 |

Each design is built per-channel (one INT8 scale per output channel; "channel-wise") and per-tensor (one scale per layer; "layer-wise"). The INT8 models take a 512 x 512 picture; the FP32 reference takes 640 x 640.

## Results at a glance (per-channel INT8, model time per image, batch 1, 4 pinned cores)

The INT8 models take a 512 x 512 picture, so each runtime also has the unquantized FP32 model at 512 px: comparing with that row shows the effect of INT8 alone, comparing with the 640 px row also includes the smaller picture.

| Runtime | Model | mAP50-95 | Retained | Size (MB) | ms per image | Speed-up vs FP32 640 px | Speed-up vs FP32 512 px |
|---|---|---|---|---|---|---|---|
| ONNX Runtime | FP32 (640 px) | 0.2811 | 100.0% | 7.9 | 28.9 | 1.00x | 0.68x |
| ONNX Runtime | FP32 (512 px) | 0.2726 | 97.0% | 7.8 | 19.6 | 1.48x | 1.00x |
| ONNX Runtime | M1 (512 px) | 0.2522 | 89.7% | 2.28 | 18.0 | 1.61x | 1.09x |
| ONNX Runtime | M2 (512 px) | 0.2694 | 95.8% | 4.80 | 15.2 | 1.90x | 1.28x |
| ONNX Runtime | M3 (512 px) | 0.2668 | 94.9% | 2.78 | 17.3 | 1.67x | 1.13x |
| TFLite | FP32 (640 px) | 0.2811 | 100.0% | 7.9 | 29.5 | 1.00x | 0.66x |
| TFLite | FP32 (512 px) | 0.2726 | 97.0% | 7.8 | 19.3 | 1.53x | 1.00x |
| TFLite | M1 (512 px) | 0.2469 | 87.8% | 2.15 | 10.7 | 2.75x | 1.80x |
| TFLite | M2 (512 px) | 0.2677 | 95.2% | 4.73 | 14.4 | 2.04x | 1.34x |
| TFLite | M3 (512 px) | 0.2643 | 94.0% | 2.67 | 11.8 | 2.51x | 1.64x |
| OpenVINO | FP32 (640 px) | 0.2811 | 100.0% | 7.9 | 24.5 | 1.00x | 0.64x |
| OpenVINO | FP32 (512 px) | 0.2726 | 97.0% | 7.8 | 15.8 | 1.55x | 1.00x |
| OpenVINO | M1 (512 px) | 0.2669 | 95.0% | 3.13 | 10.7 | 2.29x | 1.48x |
| OpenVINO | M2 (512 px) | 0.2713 | 96.5% | 5.39 | 12.7 | 1.93x | 1.25x |
| OpenVINO | M3 (512 px) | 0.2692 | 95.7% | 3.46 | 11.3 | 2.17x | 1.40x |

End to end (resize + model + box decoding) the 512 px OpenVINO M3 model takes about 17 ms against 41 ms for the FP32 baseline with the reference post-processing. These are laptop CPU figures, not NPU figures.

## Where things are

| Folder | What is in it |
|---|---|
| **`results/report.html`** | The full report: models, method, all tables and charts, studies, recommendations. |
| **`results/benchmark_summary.html`** | The short version: the six benchmarks (mAP@0.5, mAP@0.5:0.95, size, time, precision/recall/F1, memory), one section each. |
| `results/*.json`, `summary.csv` | The numbers behind the reports (accuracy, latency, memory, precision/recall, studies). `summary.csv` has one row per model. |
| `results/charts/`, `charts_summary/` | The charts as images. |
| `models/fp32/` | The original YOLOv5n (`.pt`, ONNX with pixel boxes, ONNX with boxes normalised to 0-1). |
| `models/fp32_512/` | The same network exported at 512 x 512 (the source of the INT8 models). |
| `models/onnx_int8_512/` | The six ONNX Runtime INT8 models (QDQ format). |
| `models/tflite/` | The TFLite models: `t512_*` are the INT8 ones (one `.tflite` file each; `_io8` has integer input and output), `m0_fp32` and `s512_m0_fp32` are FP32. |
| `models/ov_int8/` | The three OpenVINO INT8 models (`.xml` + `.bin`). |
| `scripts/` | Everything that builds, evaluates and benchmarks the models and writes the reports (see below). |
| `docs/` | `METHOD.md` (how it is done), `VERIFICATION.md` (every check made on the results and the corrections that followed). |
| `vendor/yolov5/` | The YOLOv5 v7.0 source, used only to export ONNX. |

The model tester that runs these models on a webcam or a video is in the folder `Real time testing` next to this one.

## Rebuilding (in order)

```
python scripts/prepare_data.py          # annotations and the 500 calibration pictures (reads the COCO val zip from Day 1)
bash scripts/build_models_512.sh        # export, quantize and score every INT8 model (ONNX Runtime, OpenVINO, TFLite)
bash scripts/run_all_benchmarks.sh      # latency, memory, post-processing, then the tables, charts and both reports
```

`build_models_512.sh` sets `YOLO_IMGSZ=512` (the model input size used by the build and evaluation scripts). `scripts/registry.py` lists every benchmarked model; `scripts/common.py` holds the data access, pre- and post-processing and the evaluation loop.

## Environments

- **Main Python** (3.11): ONNX Runtime, OpenVINO + NNCF, LiteRT (`ai-edge-litert`, runs the TFLite models), PyTorch (only for the export), OpenCV, pycocotools, matplotlib. See `requirements.txt`; NumPy must stay below 2 for the PyTorch build.
- **TensorFlow environment** (`.venv`, not in git): TensorFlow 2.15 and onnx2tf, only to convert models to TFLite.

The `onnxruntime_profile`-type dumps, raw per-image detection files, training images and the earlier experiments (ReLU activations, other input sizes, two-file TFLite pipelines) are not part of this folder; they were archived.
