"""Run a (possibly multi-stage) TFLite pipeline. Use with the TensorFlow venv.

A pipeline is a folder with stage1.tflite, stage2.tflite, ... and manifest.json. Stage k's outputs feed stage k+1's
inputs, matched by tensor shape (all boundary shapes are distinct). INT8 stages take/give float tensors at their edges, so INT8 and FP32 stages mix freely.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

THREADS = 4


class Pipeline:
    def __init__(self, folder, threads=THREADS):
        self.folder = Path(folder)
        man = json.loads((self.folder / "manifest.json").read_text())
        self.stages = []
        for name in man["stages"]:
            it = tf.lite.Interpreter(model_path=str(self.folder / name), num_threads=threads)
            it.allocate_tensors()
            self.stages.append((it, it.get_input_details(), it.get_output_details()))
        self.size_mb = man["size_mb"]

    def __call__(self, x_nchw):
        feeds = [np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1), dtype=np.float32)]   # NCHW -> NHWC
        for it, ins, outs in self.stages:
            for d in ins:                       # match by shape: the converter may reorder a stage's outputs
                v = next(f for f in feeds if tuple(f.shape) == tuple(d["shape"]))
                it.set_tensor(d["index"], v)
            it.invoke()
            feeds = [it.get_tensor(d["index"]) for d in outs]
        return feeds[0].reshape(-1, 85)
