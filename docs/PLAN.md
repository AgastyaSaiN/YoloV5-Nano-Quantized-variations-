# Day 2 Plan: YOLOv5n INT8 study for an embedded NPU target

Status: **completed.** See README.md and results/report.html for the outcome. Day 1 is frozen and was only read, never edited.

## Goal
Compare an FP32 YOLOv5n baseline against three INT8 designs, each quantized per-channel and per-tensor, in ONNX and in TFLite. Score them with industry-standard metrics. The eventual target is an i.MX8-class board with an NPU (exact board unknown, so nothing here is board-specific).

## Model
Original **YOLOv5n v7.0** (anchor-based, PANet neck, coupled Detect head). Day 1 used the newer `yolov5nu`, which is a different head. Exported with the official yolov5 export script (source downloaded as a zip, no git).

## Network parts (YOLOv5 v7.0 layer numbers)
| Part | Layers | Role |
|---|---|---|
| Backbone | 0-9 | Extracts features (outputs of 4, 6, 9 go onward) |
| Neck (PANet) | 10-23 | Combines features (outputs of 17, 20, 23 go to the head) |
| Head (Detect) | 24 | Boxes, scores, classes |

## The models (7 base, each in ONNX and TFLite)
| ID | Backbone | Neck | Head | Notes |
|---|---|---|---|---|
| M0 | FP32 | FP32 | FP32 | Baseline |
| M1 | INT8 | INT8 | INT8 | True control: everything INT8. In the yolov5 TFLite flow, box coordinates are normalised to 0-1, so unlike Day 1's ONNX control this may not collapse. We measure rather than assume. |
| M2 | INT8 | FP32 | FP32 | |
| M3 | INT8 | INT8 | FP32 | Day 1's best design |

M1-M3 are each built twice: **per-channel** (`_pc`, one scale per output channel) and **per-tensor** (`_pt`, one scale per layer). That is 1 + 6 = 7 base models.

## How mixed precision is done
- **ONNX:** one graph; FP32 layers are excluded from quantization (as in Day 1).
- **TFLite:** TFLite cannot easily leave part of a model in FP32, so the network is **split at the part boundaries** into separate files (backbone / neck / head). INT8 parts get full-integer conversion; FP32 parts stay float. A runner chains them and the benchmark times the whole chain. This also mirrors deployment: the NPU runs the INT8 part, the CPU runs the FP32 part.

## Data
| Purpose | Data | Why |
|---|---|---|
| Calibration | 500 images from COCO **train2017** | Standard practice; no overlap with the test set |
| Test | **All 5000 COCO val2017** images | Day 1 used only 1000, which was too noisy |
| Scoring | Official `instances_val2017.json` with **pycocotools** | The accepted COCO metric |

Day 1's already-downloaded val2017 images are reused by path, not copied.

## Benchmarks
| Metric | How |
|---|---|
| mAP50, mAP50-95 | pycocotools, plus AP for small / medium / large objects |
| Precision, Recall, F1 | at a fixed confidence threshold, plus best-F1 point |
| File size, compression ratio | sum of files in the pipeline vs M0 |
| Time per image | batch 1, warm-up, 200 timed runs, median and p95; preprocessing, inference, post-processing separate |
| Memory | peak process memory (RSS) in a fresh process per model |
| Confidence intervals | bootstrap over images, so we can say when two models really differ |
| Calibration-size study | 100 / 250 / 500 images on one model |
| Layer sensitivity | quantize one layer group at a time; rank by accuracy lost |

## Phases
1. Setup: yolov5 source, weights, data, tools (separate virtual environment for TensorFlow).
2. Export M0 to ONNX; build the scoring harness; reproduce the FP32 baseline mAP (expected about 0.28 mAP50-95 for YOLOv5n).
3. Build ONNX INT8 variants and benchmark them.
4. Layer sensitivity and calibration-size studies.
5. TFLite conversion (split) and benchmark.
6. Report (same plain white style as Day 1) and docs.

## Folder layout
```
Day 2/
├── README.md
├── docs/            PLAN.md, METHOD.md, BENCHMARKS.md
├── scripts/
├── vendor/yolov5/   official source (zip, no git)
├── models/{fp32,onnx_int8,tflite}/
├── data/            calibration images, annotations
├── results/         json, charts, report.html
└── .venv/           TensorFlow toolchain (kept separate to protect the Day 1 environment)
```

## Risks
- **Calibration memory.** ONNX Runtime ran out of memory at 300 images in Day 1; 500 may need streamed or chunked calibration.
- **TensorFlow tooling versions** are fragile on Windows, hence the separate environment.
- **Operator support** (YOLOv5 uses SiLU) may differ between ONNX and TFLite paths.
