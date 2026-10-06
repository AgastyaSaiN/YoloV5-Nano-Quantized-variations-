# Method

How the models were built and tested, so you can reproduce or change it.

## Pipeline

1. **Start model:** `yolov5nu.pt`, the ultralytics release of YOLOv5 nano (~5.6 MB).
2. **Export:** to FP32 ONNX, 640x640, opset 13, simplified (10.8 MB).
3. **Pre-process:** ONNX Runtime shape inference and graph clean-up.
4. **Calibrate:** run 100 COCO images through the model to measure the value range at every layer.
5. **Quantize:** static INT8 in QDQ format (weights int8 per-channel, activations uint8), which works with ONNX Runtime, TensorRT and OpenVINO.
6. **Protect the head:** every node whose name starts with `/model.24/` (the Detect layer) is passed to `nodes_to_exclude`, so it stays FP32 and untouched.
7. **Benchmark:** ultralytics validator on 1000 held-out COCO images at conf 0.001, IoU 0.7, plus 50 timed runs on the CPU.

## The five variants

| Name | Calibration method | Kept in FP32 |
|---|---|---|
| v1 | Percentile (99.999) | Detect head (66 nodes) |
| v2 | Min-Max | Detect head |
| v3 | Percentile (99.99), clips more outliers | Detect head |
| v4 | Percentile (99.999) | Detect head + the three neck output blocks (`/model.17/`, `/model.20/`, `/model.23/`) |
| control | Percentile (99.999) | Nothing |

## Verified, not assumed

For v1-v4, every one of the 19 Detect-layer convolutions (including the DFL layer) was checked against the original: weights are **FP32 and numerically identical**, with **zero Quantize/Dequantize nodes inside the head**. The control has 134 such nodes inside the head.

The head still *receives* INT8-rounded features from the layers before it. That is where the ~1% accuracy loss comes from.

## Data

- **Source:** COCO val2017 (5000 images), downloaded from images.cocodataset.org, with YOLO-format labels from the ultralytics assets release.
- **Calibration:** the first 100 images (alphabetically by file name) from a 300-image pool.
- **Test:** the **last 1000** images, with no overlap with calibration.

## Gotchas we hit

| Problem | Fix |
|---|---|
| A downloaded `yolov5n.onnx` was FP16, and the quantizer rejected it | `quantize_yolov5.py` converts FP16 to FP32 automatically |
| Calibrating on 300 images ran out of RAM (ONNX Runtime stores every layer's output) | Use 100 images |
| ONNX Runtime's `Entropy` calibration gave scales identical to Min-Max, so it isn't a real extra variant | Replaced by Percentile 99.99 |
| `pip install opencv-python` upgraded NumPy to 2.x, which breaks torch 2.3.1 ("Numpy is not available") | `numpy==1.26.4` and `opencv-python<4.12` (see `requirements.txt`) |
| Quantizing the head collapses accuracy to 0 | Keep it FP32. This is the whole point of the project. |

## Ideas for next steps

- Calibrate on your own deployment images.
- Evaluate on more images (all 5000) to separate v1-v4 with confidence.
- Time it on the actual target hardware (GPU, phone, Raspberry Pi).
- Try PyTorch FX quantization with the head left unquantized.
