"""Where does the SiLU cost sit?  The cost of an element-wise activation is proportional to its tensor size.
Lists every SiLU in layers 0-23 with its output size, sorted by size, with the cumulative share of the total.
    python scripts/activation_cost.py  -> results/activation_cost.json"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

DAY2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAY2 / "vendor" / "yolov5"))


def silu_modules(model):
    """[(name, layer_index, module)] for every SiLU in layers 0-23, in network order."""
    out = []
    for i, layer in enumerate(model.model):
        if i >= 24:
            continue
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.SiLU):
                out.append((f"{i}.{name}", i, mod))
    return out


def measure(model):
    sizes = {}
    hooks = [m.register_forward_hook(lambda mod, inp, o, n=n: sizes.__setitem__(n, o.numel())) for n, i, m in silu_modules(model)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    for h in hooks:
        h.remove()
    return sizes


if __name__ == "__main__":
    ckpt = torch.load(DAY2 / "models/fp32/yolov5n.pt", map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    sizes = measure(model)
    tot = sum(sizes.values())
    rows = sorted(sizes.items(), key=lambda kv: -kv[1])
    cum, out = 0, []
    for k, (n, s) in enumerate(rows, 1):
        cum += s
        out.append({"rank": k, "module": n, "layer": int(n.split(".")[0]), "elements": s, "cumulative_share": cum / tot})
    (DAY2 / "results" / "activation_cost.json").write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} SiLU activations, {tot / 1e6:.1f} M elements in total")
    for k in (1, 3, 5, 8, 12, 16, 20, 25, 30, 40, 57):
        print(f"  top {k:2d} activations by size -> {out[k - 1]['cumulative_share'] * 100:5.1f}% of all SiLU work (layers used: {sorted({r['layer'] for r in out[:k]})})")
