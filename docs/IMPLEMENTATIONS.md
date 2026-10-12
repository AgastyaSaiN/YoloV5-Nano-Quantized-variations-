# Implementations by period

The work is grouped into three periods. Each period is described below; the charts and tables then show the same benchmarks for all three, older periods in lighter shades. The Detect head is untouched in every model of every period.

## Day 2 implementations

- FP32 YOLOv5n baseline and three INT8 designs: M1 everything INT8, M2 backbone INT8 with neck and head FP32, M3 backbone and neck INT8 with the head FP32.
- Each design in two granularities (per-channel and per-tensor) and two formats: ONNX (ONNX Runtime, percentile calibration) and TFLite (full-integer, min-max calibration, split into separate files so part of the network can stay FP32).
- Box coordinates normalised to 0-1 so that the all-INT8 model works. Calibration on 500 train2017 images; testing on all 5000 val2017 images with the official COCO evaluation.
- Extra studies: layer sensitivity, calibration-set size, 10-fold statistical comparison.

## Day 3 implementations

- Audit and corrections: pinned cores for timing, an idle machine, matched calibration methods for the ONNX versus TFLite comparison, and official COCO matching for precision and recall.
- OpenVINO quantization (NNCF) of the same designs: the smallest files and a faster INT8 runtime on this CPU.
- ReLU-activation models: R55 (12 of 57 activations replaced, 55% of the activation work) and R80 (29 of 57, 81%), fine-tuned with the Detect head frozen. They are faster but less accurate (see the tables and the speed-versus-accuracy chart).

## Current stage

- Reduced input size: the original network exported at 512, 448 and 416 pixels and quantized with the same OpenVINO procedure, no retraining. At 512 pixels M3 is 2.14x faster than the 640 px FP32 model at 95.7% of its accuracy; at 448 pixels 2.73x at 92.2%.
- Compiled post-processing with identical output (12.5 ms down to 3.1 ms per image at confidence 0.25).
- Tried and not adopted: fine-tuning at the smaller size, with or without ReLU, scored below the un-retrained network at the same size.
- Not yet tried: channel pruning and quantization-aware training.

Charts and tables of the same benchmarks across the three periods: results/report.html and results/benchmark_summary.html.
