"""Exact (output-identical) graph rewrites that make the network faster on CPU runtimes. Weights are not changed and the
Detect head (layer 24) is never touched.

rewrite_resize_to_expand(): YOLOv5's 2x nearest-neighbour upsample (ONNX Resize) is replaced by Reshape -> Expand ->
    Reshape, i.e. plain pixel replication. ONNX Runtime's Resize kernel costs about 2.7 ms per upsample on this CPU;
    the reshape/expand form is a memory copy.

    python scripts/graph_opt.py <in.onnx> <out.onnx>
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper as nh, shape_inference


def rewrite_resize_to_expand(model):
    inferred = shape_inference.infer_shapes(model)
    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in list(inferred.graph.value_info) + list(inferred.graph.input)}
    new_nodes, count = [], 0
    for n in model.graph.node:
        if n.op_type == "Resize" and n.name.startswith(("/model.11/", "/model.15/")):
            mode = next((a.s.decode() for a in n.attribute if a.name == "mode"), "nearest")
            assert mode == "nearest", mode
            N, C, H, W = shapes[n.input[0]]
            tag = n.name.replace("/", "_")
            consts = {}
            for key, shp in (("s1", [N, C, H, 1, W, 1]), ("s2", [N, C, H, 2, W, 2]), ("s3", [N, C, 2 * H, 2 * W])):
                nm = f"{tag}_{key}"; consts[key] = nm
                model.graph.initializer.append(nh.from_array(np.array(shp, dtype=np.int64), nm))
            a, b = f"{tag}_r1", f"{tag}_e"
            new_nodes += [helper.make_node("Reshape", [n.input[0], consts["s1"]], [a], name=n.name + "_reshape1"),
                          helper.make_node("Expand", [a, consts["s2"]], [b], name=n.name + "_expand"),
                          helper.make_node("Reshape", [b, consts["s3"]], list(n.output), name=n.name + "_reshape2")]
            count += 1
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return count


if __name__ == "__main__":
    m = onnx.load(sys.argv[1])
    print("rewrote", rewrite_resize_to_expand(m), "Resize nodes")
    onnx.save(m, sys.argv[2])
