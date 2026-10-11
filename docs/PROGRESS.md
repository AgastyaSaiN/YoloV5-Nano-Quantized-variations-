# Progress, stage by stage

Each stage was prompted by what the previous one showed. One representative design (M3: backbone and neck INT8, Detect head FP32) is followed through the stages; the Detect head is untouched at every stage. All numbers are on all 5000 COCO val2017 images.

| Stage | Model shown | mAP50 | mAP50-95 | Retained | Size / compression | ms per image | Speed-up | P / R / F1 @0.25 | Peak memory added (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | FP32 baseline (ONNX Runtime) | 0.461 | 0.2811 | 100% | 7.92 MB / 1.0x | 28.6 | 1.00x | 0.691 / 0.460 / 0.552 | 62 |
| 1 | First INT8 models: M3 per-channel (ONNX Runtime) | 0.452 | 0.2735 | 97.3% | 2.92 MB / 2.7x | first run, unpinned (overstated) | about 1.5x (overstated) | 0.707 / 0.442 / 0.544 | 66 |
| 2 | Audit and corrections: M3 per-channel, re-measured (ONNX Runtime) | 0.452 | 0.2735 | 97.3% | 2.92 MB / 2.7x | 24.9 | 1.15x | 0.707 / 0.442 / 0.544 | 66 |
| 3 | OpenVINO quantization: M3, NNCF (OpenVINO) | 0.453 | 0.2748 | 97.7% | 0.95 MB / 8.3x | 14.9 | 1.44x | 0.698 / 0.451 / 0.548 | 115 |
| 4 | ReLU in the largest layers (R55): R55 M3, NNCF (OpenVINO) | 0.436 | 0.2600 | 92.5% | 0.90 MB / 8.8x | 12.6 | 1.71x | 0.638 / 0.458 / 0.533 | 109 |
| 5 | More ReLU (R80): R80 M3, NNCF (OpenVINO) | 0.421 | 0.2490 | 88.6% | 0.82 MB / 9.7x | 11.5 | 1.88x | 0.643 / 0.441 / 0.523 | 102 |

## Stage 1: First INT8 models

**What we did.** Built the three designs (M1 everything INT8, M2 backbone INT8, M3 backbone and neck INT8; Detect head FP32 in M2 and M3), each per-channel and per-tensor, in ONNX and TFLite. Scored on all 5000 COCO val2017 images.

**What came out.** Accuracy held up well (M2 98%, M3 97% of FP32 in ONNX), files shrank 2.7 to 3.4x, but the speed-up was small. ONNX and TFLite gave different accuracy for the same design.

## Stage 2: Audit and corrections

**What we did.** Questioned every number. Found and fixed: unpinned threads that inflated the FP32 baseline by about 50%, a timing run taken while training was running, a calibration cache that froze value ranges, and a precision/recall matcher that ignored crowd regions. Re-measured everything on an idle machine with four pinned cores.

**What came out.** No model changed, but the numbers became trustworthy. The ONNX-versus-TFLite gap turned out to be the calibration method (percentile versus min-max): with the method matched, the formats agree within about 0.2 mAP points.

## Stage 3: OpenVINO quantization

**What we did.** Quantized the same network with OpenVINO and its NNCF tool, which handles activations better than the ONNX Runtime QDQ format.

**What came out.** Faster than ONNX Runtime and the smallest files, with 97.7% of the accuracy. Still limited by the SiLU activation, which stays a separate sigmoid and multiply in INT8.

## Stage 4: ReLU in the largest layers (R55)

**What we did.** Replaced the 12 largest SiLU activations (about 55% of the activation work: layers 0 to 4 and 17) with ReLU and fine-tuned the network with the Detect head frozen and unchanged (verified parameter by parameter).

**What came out.** ReLU fuses into the INT8 convolution, so the INT8 speed-up rose clearly. About 5 points of accuracy were given up for it.

## Stage 5: More ReLU (R80)

**What we did.** Replaced 29 of the 57 activations (81% of the activation work) and fine-tuned for 30 epochs.

**What came out.** Faster again, and the all-INT8 R80 model (M1) passes 2x; but each extra step costs about 3 to 4 accuracy points, so this is the fast end of the trade-off.
