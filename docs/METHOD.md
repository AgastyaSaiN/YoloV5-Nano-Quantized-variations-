# Day 2 Method

## Pipeline
1. **Model:** original YOLOv5n v7.0 (`yolov5n.pt`), exported with the official `export.py` (opset 13, 640x640, FP32) from the downloaded source archive.
2. **Box normalisation:** `make_normalized_model.py` divides six decode constants in the Detect head (three strides for xy, three anchor grids for wh) by 640, so the box output is 0-1. No extra graph nodes. Verified against the original (maximum difference 6e-5 pixels). Evaluation multiplies boxes by 640 after inference.
3. **ONNX INT8** (`build_onnx_variants.py`): ONNX Runtime static quantization, QDQ format, weights int8 symmetric, activations uint8, percentile calibration (99.999) on 500 train2017 images. FP32 layers are excluded by node-name prefix: backbone `/model.0-9/`, neck `/model.10-23/`, head `/model.24/`. Per-channel and per-tensor via `per_channel=True/False`.
4. **TFLite INT8:** the normalised ONNX is split into backbone / neck / head parts (`split_onnx.py`; chained output verified identical to the full model), each converted to a TF SavedModel with onnx2tf (`convert_parts_onnx2tf.sh`), then to TFLite with the TF converter (full-integer, 500 calibration images, min-max, strict INT8 ops). FP32 stages use the float32 conversion. `tflite_runner.py` chains the stages, matching tensors by shape.
5. **Accuracy** (`common.py`, `evaluate_*.py`): square 640 letterbox, confidence 0.001, NMS IoU 0.6, up to 300 detections, multi-label, as in the official `val.py`; scored with `pycocotools` on all 5000 val2017 images.
6. **Precision, recall, F1** (`compute_prf.py`): official COCO matching (greedy by score at IoU 0.5, crowd regions ignored, up to 300 detections per image), micro-averaged over classes and images; confidence sweep 0.05-0.95 for P/R/F1 at 0.25 and the best F1.
7. **Latency** (`benchmark_latency.py`): batch 1, 4 threads, fresh process per measurement; 20 warm-up and 100 timed runs per round; all 14 models interleaved over 9 rounds in shuffled order; figure = median of per-round medians; round spread reported.
8. **Memory** (`benchmark_memory.py`): fresh process per model; baseline resident memory recorded after the runtime is imported and one input is created; memory added at load and at peak during 35 inferences (2 ms sampler); three repeats, medians reported (`aggregate_memory.py`).
9. **Profile** (`profile_ort.py`): ONNX Runtime per-node kernel times (median over 40 runs), grouped by network region and operation type.
10. **Statistics** (`fold_stats.py`, `paired_comparisons.py`): 5000 images split into 10 disjoint folds; per-fold mAP and paired differences with 95% t-intervals.
11. **Studies:** layer sensitivity (`layer_sensitivity.py`, `sensitivity_qdq_check.py`), calibration-set size (`calib_size_study.py`), calibration method (`calib_method_study.py`).

## The designs
| Name | Calibration | Kept in FP32 |
|---|---|---|
| M1 | percentile | nothing |
| M2 | percentile | neck + head |
| M3 | percentile | head |

Each is built per-channel and per-tensor, in ONNX and in TFLite.

## Data
- **Calibration:** 500 images, seeded random sample of COCO train2017 (`prepare_data.py`).
- **Test:** all 5000 COCO val2017 images with the official annotations (read from Day 1's download in place).

## Problems encountered (and fixes)
See [VERIFICATION.md](VERIFICATION.md) for the checks that led to corrections. Environment and tooling problems:

| Problem | Fix |
|---|---|
| ONNX Runtime calibration ran out of RAM (it keeps every image's outputs) | Feed the calibrator 15 images at a time (it merges incrementally) |
| Min-max calibration: this ONNX Runtime version's own memory cap discards data without recording it | Same chunking approach for the min-max calibrator |
| Calibration was repeated for every variant | Ranges do not depend on exclusions or granularity: cached per (model, N, method) |
| All-INT8 model scored 0.000 | Box output in pixels shares one INT8 scale with 0-1 scores; normalise boxes to 0-1 |
| `onnx2tf` tried to download a sample file | Provide a local placeholder file in its working directory |
| `onnx` 1.19 conflicts with TensorFlow 2.15's `ml_dtypes` | Pin `onnx==1.16.2`, `onnxruntime==1.18.1` in the TensorFlow environment |
| A TFLite INT8 stage returned outputs in a different order | Match stage inputs to previous outputs by tensor shape |
| A collapsed model has zero detections and `loadRes` fails | Treat an empty detections file as AP 0 |

## Reproduce
```bash
pip install -r requirements.txt            # main environment; TensorFlow environment in Day 2/.venv (see requirements.txt)
python scripts/prepare_data.py
python scripts/make_normalized_model.py
python scripts/build_onnx_variants.py                        # 6 ONNX variants
python scripts/build_onnx_variants.py m1_all_int8_pc m1_all_int8_pt --norm
python scripts/split_onnx.py
bash scripts/convert_parts_onnx2tf.sh                        # ONNX parts -> SavedModels
.venv/Scripts/python.exe scripts/build_tflite_variants.py
bash scripts/run_onnx_benchmarks.sh                          # accuracy, ONNX
bash scripts/run_tflite_benchmarks.sh                        # accuracy, TFLite
bash scripts/run_remeasurements.sh                             # P/R/F1, quantized-layer check, memory x3, latency x9 rounds (idle machine)
python scripts/aggregate_memory.py
python scripts/fold_stats.py && python scripts/paired_comparisons.py
python scripts/profile_ort.py
python scripts/layer_sensitivity.py && python scripts/calib_size_study.py && python scripts/calib_method_study.py
python scripts/make_summary.py && python scripts/make_charts.py && python scripts/make_report.py && python scripts/make_benchmark_summary.py
```
