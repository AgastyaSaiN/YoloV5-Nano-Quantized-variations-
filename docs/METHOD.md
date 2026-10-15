# Method

## Pipeline
1. **Model:** original YOLOv5n v7.0 (`models/fp32/yolov5n.pt`), exported with the official `export.py` (opset 13, FP32) from the vendored source: 640 x 640 for the FP32 reference, 512 x 512 for the INT8 models (`export_onnx.py`).
2. **Box normalisation** (`make_normalized_model.py`, `export_onnx.py`): six decode constants in the Detect head (three strides for xy, three anchor grids for wh) are divided by the input size, so boxes come out as fractions of the picture. Without this the all-INT8 model scores zero: boxes up to 640 and confidence scores below 1 cannot share one INT8 scale. The head weights are not touched.
3. **ONNX Runtime INT8** (`build_onnx_variants.py`): static quantization, QDQ format, weights int8 symmetric, activations uint8, percentile calibration, 500 calibration pictures. FP32 parts are excluded by node name (M2: neck + head, M3: head). Per-channel and per-tensor.
4. **OpenVINO INT8** (`quantize_openvino.py`): NNCF post-training quantization, mixed preset, the same 500 pictures; M2/M3 exclude the neck/head by name pattern.
5. **TFLite INT8** (`split_onnx.py`, `convert_parts_onnx2tf.sh`, `build_tflite_single.py`): the ONNX model is converted with onnx2tf, then quantized with the TensorFlow Lite converter (full integer, min-max calibration on the same 500 pictures, per-channel weights by default, per-tensor with a switch). For M2 and M3 the layers that stay float are excluded with the quantization debugger's `denylisted_nodes`, so each design is one `.tflite` file; the excluded operations are found from the graph structure (ancestors of the backbone / neck outputs, located by value matching against separately converted parts). `--int-io` builds M1 with int8 input and output. TFLite models run with LiteRT (`tflite_runner.py`).
6. **Accuracy** (`common.py`, `evaluate_*.py`): letterbox to the model's input size, confidence 0.001, NMS IoU 0.6, up to 300 detections, multi-label, as in the official `val.py`; all 5000 val2017 images, official `pycocotools`.
7. **Precision, recall, F1** (`compute_prf.py`): official COCO matching (greedy by score at IoU 0.5, crowd regions ignored, up to 300 detections per image), micro-averaged over classes and images, at confidence 0.25.
8. **Latency** (`benchmark_latency.py`): batch 1, 4 threads pinned to four distinct physical cores, high priority, fresh process per measurement, 20 warm-up and 100 timed runs per round, 9 rounds with all models interleaved in shuffled order; headline = median of the per-round medians. Model time only; pre- and post-processing are timed separately (`benchmark_postprocess.py`).
9. **Memory** (`benchmark_memory.py`, `aggregate_memory.py`): fresh process per model; resident memory sampled every 2 ms; the figure is the increase over a process that has only loaded the runtime; median of 3 runs.
10. **TFLite runtime check** (`probe_tflite_runtime.py`): the same `.tflite` files timed with TensorFlow's interpreter and with LiteRT.
11. **Studies** (results in `results/*.json`, run on the 640 px models): layer sensitivity (`layer_sensitivity.py`, `sensitivity_qdq_check.py`), calibration-set size and method (`calib_size_study.py`, `calib_method_study.py`, `calib_matched_comparison.py`), a per-operation profile (`profile_ort.py`), and 10-fold paired statistics (`fold_stats.py`, `paired_comparisons.py`). The 640 px INT8 models these were run on are archived; `build_onnx_variants.py` rebuilds them from `models/fp32/yolov5n_norm.onnx`.
12. **Reports** (`make_summary.py`, `make_charts.py`, `make_report.py`, `make_benchmark_summary.py`, with `eras.py` and `extra_sections.py` for the cross-runtime sections): everything is generated from `results/*.json`; no number is typed into a report by hand.

## The designs
| Name | Kept in FP32 |
|---|---|
| M1 | nothing |
| M2 | neck + head |
| M3 | head |

Each is built per-channel and per-tensor in ONNX Runtime and TFLite, and per-channel (NNCF default) in OpenVINO.

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
