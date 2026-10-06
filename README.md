# YOLOv5 Nano, Shrunk to INT8, with the Detection Head Left Alone

We compressed the **YOLOv5 nano** object detector (quantization to 8-bit integers) and deliberately **did not touch the final Detect layer**.
The result is a model that is **about half the size, 10-20% faster, and only ~1% less accurate**.

![Accuracy](results/charts/1_accuracy.png)

## The result in 30 seconds

| | Original | **Best compressed (v1)** |
|---|---|---|
| Accuracy (mAP50-95) | 0.335 | **0.332** (-0.9%) |
| File size | 10.8 MB | **5.8 MB** (46% smaller) |
| Time per image (CPU) | 56 ms | **45 ms** (19% faster) |
| Detection layer | FP32 | **FP32, identical to the original** |

**Open [results/report.html](results/report.html)** in any browser for a printable, plain-English report with the charts.

**Use this file:** [`models/int8/v1_percentile_head.onnx`](models/int8/v1_percentile_head.onnx)

When we also compressed the detection layer (the "control" model), accuracy fell to **0.000**. Keeping that layer untouched is what makes this work. Details are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Plain-English glossary

| Term | Meaning |
|---|---|
| **Quantization** | Storing the model's numbers as small whole numbers (8-bit) instead of precise decimals (32-bit). Smaller and faster, slightly less precise. |
| **FP32 / INT8** | The precise decimal format / the small whole-number format. |
| **Detect head** | The last layer, which turns the model's internal features into actual boxes, labels and confidence scores. It is sensitive to rounding, so we keep it FP32. |
| **Calibration** | Showing the model some sample images so quantization can pick sensible number ranges. |
| **mAP** | The standard accuracy score for detectors (0 to 1, higher is better). |

## What's in this folder

```
.
├── README.md                  <- you are here
├── requirements.txt
├── docs/
│   ├── BENCHMARKS.md          <- every benchmark number explained for beginners
│   └── METHOD.md              <- exactly how the variants were built, and the gotchas
├── models/
│   ├── original/              <- yolov5nu.pt / .onnx (the untouched FP32 model)
│   └── int8/                  <- the 5 compressed models (v1..v4 + control)
├── results/
│   ├── report.html            <- the report: charts, tables, explanation
│   ├── benchmark_results.json <- raw scores
│   ├── charts/                <- the 3 charts (with titles)
│   ├── charts_for_report/     <- same charts without titles, embedded in report.html
│   └── logs/
├── scripts/
│   ├── prepare_data.py        <- downloads COCO val2017, splits calibration / test images
│   ├── quantize_yolov5.py     <- quantize ANY yolov5 ONNX, keeping the head FP32
│   ├── make_variants.py       <- builds v1..v4 + control
│   ├── benchmark.py           <- scores accuracy + speed + size
│   ├── make_charts.py         <- draws the charts
│   └── make_report.py         <- builds results/report.html
├── data/coco/                 <- calibration + test images (large; safe to delete, re-creatable)
└── archive/first_attempt/     <- earlier quick experiments, kept for reference
```

## Re-run everything

```bash
pip install -r requirements.txt
python scripts/prepare_data.py     # ~830 MB download (COCO val2017 + labels)
python scripts/make_variants.py    # build the INT8 models (~15 min)
python scripts/benchmark.py        # score them (~30 min on CPU)
python scripts/make_charts.py
```

To quantize your own YOLOv5 ONNX model with the head kept in FP32:

```bash
python scripts/quantize_yolov5.py --weights your_model.onnx --calib-dir path/to/100_images
```

## Honest limitations

- Tested on **1000 COCO images**. Differences under ~0.5% between v1-v4 are noise.
- Speed was measured on **one CPU** using ONNX Runtime. Results on GPUs, phones or other runtimes will differ.
- Calibration used 100 COCO images. For your own use case, calibrate with your own typical images.
- This is the ONNX route. PyTorch-native quantization was not done.
