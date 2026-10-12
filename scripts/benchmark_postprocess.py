"""End-to-end time per image: preprocessing + model + post-processing, with the reference numpy post-processing and the compiled-NMS version.
Checks first that both give the same detections on 100 val images, then times them (the model is the FP32 ONNX baseline; post-processing cost
does not depend on the model's precision).  python scripts/benchmark_postprocess.py -> results/postprocess.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import DAY2, ValImages, postprocess, postprocess_fast, preprocess  # noqa: E402

so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(str(DAY2 / "models/fp32/yolov5n.onnx"), so, providers=["CPUExecutionProvider"])
data = ValImages()
items = []
for i in range(100):
    img = data.read(i)[1]
    x, r, pad = preprocess(img)
    items.append((x, r, pad, img.shape[:2], sess.run(None, {"images": x})[0][0]))

same = 0
for x, r, pad, shp, pred in items:
    a, b = postprocess(pred, r, pad, shp), postprocess_fast(pred, r, pad, shp)
    same += len(a) == len(b) and np.allclose(a, b, atol=1e-3)
print(f"identical detections on {same}/100 images")

res = {"identical_images_of_100": same}
for conf in (0.001, 0.25):
    common.CONF_THRES = conf
    for name, fn in (("numpy", lambda *a: postprocess(*a)), ("compiled", lambda *a: postprocess_fast(*a, conf_thres=conf))):
        for it in items[:10]:
            fn(it[4], it[1], it[2], it[3])
        t = []
        for k in range(3):
            for it in items:
                t0 = time.perf_counter(); fn(it[4], it[1], it[2], it[3]); t.append(time.perf_counter() - t0)
        res[f"post_ms_{name}_conf{conf}"] = float(np.median(t) * 1e3)
        print(f"post-processing {name:9s} conf {conf}: {res[f'post_ms_{name}_conf{conf}']:.2f} ms")
t = []
for it in items:
    img = data.read(0)[1]
    t0 = time.perf_counter(); preprocess(img); t.append(time.perf_counter() - t0)
res["pre_ms"] = float(np.median(t) * 1e3)
print(f"preprocessing: {res['pre_ms']:.2f} ms")
(DAY2 / "results" / "postprocess.json").write_text(json.dumps(res, indent=2))
