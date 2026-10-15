# Verification and corrections
> Some entries below refer to models and experiments that are no longer in this folder (earlier input sizes, ReLU-activation variants, two-file TFLite pipelines); they were archived and the entries are kept as the record of what was checked and corrected.

Every suspicious pattern in the results was tested before the reports were finalised. This page lists each check, what it found, and what was changed. Items marked **corrected** changed numbers or claims in the reports.

| # | Check | Finding | Outcome |
|---|---|---|---|
| 1 | FP32 baseline against the published YOLOv5n figure | mAP50-95 0.2811 against 0.28 published | Pipeline confirmed |
| 2 | ONNX FP32 against TFLite FP32 | mAP differs only at the 8th decimal; detection records differ in the last digits, as expected for two engines running a lossless conversion | Conversion confirmed lossless |
| 3 | Chained split TFLite parts against the full model | Output differs by 0.0 | Split confirmed exact |
| 4 | Box-normalised model against the original | Maximum difference 6e-5 pixels | Normalisation confirmed exact |
| 5 | Is per-channel / per-tensor really applied? | ONNX: per-channel models have a scale vector on all 57-60 conv weights, per-tensor models a single scale. TFLite: per-channel scales on 57 weights, none for per-tensor | Confirmed in both formats |
| 6 | Are the TFLite INT8 stages truly integer? | INT8 stages hold 165-346 int8 tensors and only 2-4 float tensors (their inputs/outputs); the FP32 head stage is pure float | Confirmed |
| 7 | Precision, recall, F1 against the official COCO matcher | **The first implementation ignored COCO's crowd regions and counted detections inside them as false positives**, understating precision by about 0.05 (FP32: 0.645 against 0.691). Recall was identical (0.4598). The best-F1 threshold also showed a suspicious constant value of 0.25 for every model | **Corrected.** P, R and F1 are now computed with the official matching rules (`compute_prf.py`); the best-F1 threshold now varies (0.20 to 0.25) |
| 8 | Memory benchmark | **The first test kept 50 pre-processed images (246 MB) in RAM while measuring, so FP32 and INT8 looked identical.** | **Corrected.** Rebuilt (`benchmark_memory.py`): one input, baseline taken before loading, 2 ms peak sampler, three repeats. Run-to-run spread is about 1 MB (up to 7 MB for TFLite FP32), and real differences appear |
| 9 | Latency benchmark | **A single pass is unreliable.** The same FP32 file measured 27 ms and 37 ms in two sessions (laptop CPU frequency and temperature drift) | **Corrected.** Models are interleaved over 9 rounds in shuffled order; the median of per-round medians and the round-to-round spread are reported (`benchmark_latency.py`). In every round each ONNX INT8 model beat FP32 |
| 10 | Why is M2 faster than M3, and why is INT8 not faster still? | ONNX profile (`profile_ort.py`): INT8 convolutions are faster, but the SiLU activation becomes separate integer kernels (5.3 ms against 0.5 ms in FP32 where it is fused), and the INT8 neck is slower than the FP32 neck | Explained with measurements |
| 11 | Calibration-size study | **Calling the cache-patch installer repeatedly in one process made N=100 and N=250 reuse N=50's ranges (identical mAP).** | **Corrected.** Each size runs in its own process; the cached ranges were checked to converge toward the 500-image ranges (median difference 0.92%, 0.55%, 0.19% for N=50, 100, 250) |
| 12 | Layer-sensitivity study | **The upsample layers (11, 15) received no quantization when isolated (0 quantize/dequantize operations), so their "0% loss" meant nothing.** Concat layers received one quantize/dequantize pair | **Corrected.** Those layers are drawn hatched as "not quantized", and the claim that they "cost nothing" was removed |
| 13 | Why does ONNX score higher than TFLite? | Rebuilding ONNX M3 per-channel with min-max calibration (TFLite's method) gives 0.2676, against 0.2677 for TFLite and 0.2735 with percentile | Calibration method explains essentially all of the gap |
| 14 | Are accuracy differences real? | 10-fold paired comparison: every INT8 model differs from FP32, and per-channel beats per-tensor in all six pairings | Confirmed |
| 15 | Is the Detect head untouched in the FP32-head designs? (`check_head_untouched.py`) | M2 and M3, per-channel and per-tensor, ONNX and TFLite: all 3 head convolution weight tensors are bit-identical to the original FP32 weights and the head holds 0 quantize/dequantize operations. M1 (head INT8 by design): 0 of 3 identical, 100 quantize/dequantize operations in the head (ONNX), INT8 weights in TFLite | Confirmed. Note that the head still receives INT8-rounded features from the layers before it, and in the TFLite designs and the normalised model its decode constants are divided by 640 (an exact rescaling, reversed after inference) |

## Remaining limitations
- No NPU or board was used; timing and memory are desktop-CPU figures.
- Layer sensitivity uses the first 1000 validation images and one layer at a time.
- Latency is noisy (rounds of the same model differ by up to about 27%); small gaps between INT8 variants should not be interpreted.
- Quantization-aware training was not attempted: it needs the COCO training set (about 18 GB) and substantial training time. The measured accuracy loss (1.6 to 9%) leaves that as a possible future step.


## Second round of corrections

- **Pinned cores.** Unpinned threads landed on hyper-threaded siblings and inflated the FP32 baseline by about 50%. All timings now use four distinct physical cores.
- **Measurement during training.** A latency run taken while training was running gave an FP32 baseline of 63 ms (about 25 ms when idle); it was discarded. Final timings were taken on an idle machine on AC power.
- **Calibration cache bug.** A cache of min-max ranges froze the ranges after the first chunk of images; the cache was removed for min-max and the ONNX-versus-TFLite comparison was redone (differences now -0.18 to +0.10 mAP points).
- **OpenVINO/NNCF.** Quantizing with NNCF gave no speed advantage over ONNX Runtime QDQ for the SiLU models beyond the runtime itself.
- **Hybrid models.** Training is resumable; after every run all Detect-head parameters were compared with the original and are identical.
- **Hardware resets.** The laptop reset unexpectedly several times during long jobs (Windows event 41, no crash dump); jobs were made resumable and short measurements repeated.

## Third round

- **Post-processing is part of the time.** The reference numpy non-maximum suppression took 26.6 ms (threshold 0.001) and 12.5 ms (threshold 0.25) against about 11 to 17 ms for the INT8 model. A compiled routine gives identical detections on 100 of 100 images and takes 5.9 and 3.1 ms (`scripts/benchmark_postprocess.py`).
- **Smaller input.** The original network exported at 512, 448 and 416 pixels, quantized with the same OpenVINO procedure with calibration images letterboxed to the same size, scored on all 5000 images. Speed-ups are quoted against the 640 px FP32 model.
- **Negative results kept in the report.** R55 fine-tuned at 512 px and the plain network fine-tuned at 448 px with the Detect head frozen scored below the un-retrained network at the same size, so they are reported but not recommended.
- **Cross-session timing.** Latency for the OpenVINO models was re-measured together in one session so that speed-ups are computed within a single run. Absolute milliseconds differ slightly between sessions (laptop thermal state); ratios within a session are stable.
- **Head check.** The Detect head was compared parameter by parameter with the original after every fine-tune (R55 at 640, R80, R55 at 512, 448 fine-tune): identical.

## Current-stage ONNX Runtime and TFLite models (512 px)

- Rebuilt the six ONNX Runtime and six TFLite INT8 variants at 512 x 512 input from the normalised-box FP32 export, scored on all 5000 val images, timed against the 640 px FP32 baselines of the same format (interleaved, 9 rounds), memory measured 3 times.
- **Conversion fault found and fixed.** The first TFLite M2 pipeline at 512 px scored 83.5% of FP32 although its backbone is the same as M3's (93.6%). A float-only check (backbone + neck/head stages chained, no quantization) differed from the full model, while backbone-neck + head matched exactly. Cause: onnx2tf cannot tell NCHW from NHWC for the square, 64-channel (1 x 64 x 64 x 64) input of the neck+head stage and left it in NCHW order. A layout hint (`input_perm`) in the pipeline manifest, applied by `scripts/tflite_runner.py`, restores the order; the M2 pipelines then scored 95.5% and 92.7%. The other pipelines were verified unaffected (float chain identical to the full model).
- The Day 2 bars in the period charts keep the timing as first measured; the main tables use the later re-measurement of the same models (a few percent faster).

## OpenVINO file sizes (correction)

- **Fault.** `scripts/evaluate_ov.py` recorded the size of the OpenVINO model as the size of the `.xml` graph file only; the weights are in the separate `.bin` file. The OpenVINO models therefore looked about four times smaller than they are (for example 0.96 MB instead of 3.27 MB for M1) and the compression ratios quoted for them (8 to 9x) were wrong. M2 even appeared smaller than M1, which cannot be true.
- **Fix.** The size is now `.xml` plus `.bin`, for every OpenVINO model (`results/accuracy_ov.json` recomputed). Real sizes: M1 3.27 MB, M2 5.53 MB, M3 3.60 MB at 640 px (against 7.92 MB FP32), i.e. about 2.2 to 2.4x for M1/M3 and 1.4x for M2. In every runtime M1 is now the smallest and M2 the largest INT8 design.
- **Other checks of the same kind.** ONNX Runtime sizes are single files and TFLite sizes are the sum of the stage files; both were confirmed against the files on disk. The ordering of time and memory between designs differs between runtimes; the report explains each case and marks the explanations that were not verified.

## TFLite rebuilt the standard way, and the runtime found to be the bottleneck

- **Sources checked:** LiteRT post-training quantization, the quantization debugger (selective quantization with `denylisted_nodes`) and the 8-bit quantization specification. The recipe: representative dataset with inference preprocessing, int8 symmetric weights (per-channel by default, per-tensor by switch), int8 per-tensor activations, float layers excluded inside one file, integer input/output for integer-only targets.
- **Single-file models.** M1, M2, M3 x per-channel / per-tensor were rebuilt at 512 px as one `.tflite` file each (`scripts/build_tflite_single.py`). Which operations stay float is found from the graph structure (ancestors of the backbone / neck output tensors, located by value matching against separately converted parts), because the converted graph has no layer names. Checked on M3: 57 INT8 convolutions, 3 float convolutions (the Detect head). Scores are unchanged within rebuild noise (M2 95.2 / 92.8%, M3 94.0 / 90.5% of FP32; before 95.5 / 92.7 and 93.6 / 90.1).
- **Calibration size** (20 to 500 images, M3 per-channel, first 500 val images): 0.3009 to 0.3091 with no trend, so calibration size is not the lever; TFLite offers only min-max calibration.
- **Runtime.** The same `.tflite` files run 1.4 to 1.7 times faster in the current LiteRT interpreter (ai-edge-litert 2.3.0) than in TensorFlow 2.15's built-in one (`results/tflite_runtime_probe.json`); the float model is unchanged. Scores are identical (M3 per-channel on 500 images: 0.3079 with both). All TFLite timing and memory figures were re-measured with LiteRT, including the earlier 640 px models.
- **Integer input/output.** M1 with int8 input and output scores the same as M1 with float input/output (87.8% per-channel) and takes 7.9 to 8.1 ms of model time against 9.5 to 9.7 ms. In the benchmark runner it looked slower (20 ms) only because the runner quantizes the picture in Python (about 8 ms); the report shows the model-only time.
- **Environment note.** Installing `ai-edge-litert` pulled NumPy 2.x into the main environment; NumPy was pinned back to 1.26.4 (needed by the PyTorch build) and everything was re-checked. pip prints a harmless version complaint about `ml-dtypes`.
