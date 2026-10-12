"""Fine-tune the ReLU YOLOv5n (SiLU -> ReLU in layers 0-23) by DISTILLATION from the original SiLU model.

* The Detect head (layer 24) is FROZEN and is never modified; at the end its weights are verified to be bit-identical
  to the original model's.
* Teacher: original YOLOv5n (SiLU). Student: same network with ReLU, initialised from the teacher's weights.
* Loss (no labels needed): (1) the student's raw head outputs - passed through the same frozen head - must match the
  teacher's (soft BCE on objectness/class logits + smooth-L1 on box logits); (2) intermediate feature maps (layers
  4, 6, 9, 13, 17, 20, 23) must match, normalised by the teacher's feature variance.
* Data: COCO train2017 images only (never val2017), random scale / crop / flip / colour jitter.

Run with the GPU environment:
    .venv_train\\Scripts\\python.exe scripts\\finetune_relu.py --epochs 12
Output: models/relu/relu_trained.pt (YOLOv5 checkpoint format), results/finetune_log.json
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DAY2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAY2 / "vendor" / "yolov5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
FEATURE_LAYERS = [4, 6, 9, 13, 17, 20, 23]
HOLDOUT = 500


class Images(torch.utils.data.Dataset):
    def __init__(self, files, size=640, train=True, label_dir=None):
        self.files, self.size, self.train = files, size, train
        self.label_dir = Path(label_dir) if label_dir else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(str(self.files[i]))
        if img is None:                                                 # unreadable / still being downloaded
            return self[(i + 1) % len(self.files)]
        h, w = img.shape[:2]
        r = self.size / max(h, w)
        flipped = False
        if self.train:
            r *= random.uniform(0.6, 1.4)                               # scale jitter
            if random.random() < 0.5:
                img = img[:, ::-1]; flipped = True
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.float32)   # colour jitter
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-0.015, 0.015) * 180) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.6, 1.4), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.3), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        nh, nw = max(int(round(h * r)), 8), max(int(round(w * r)), 8)
        img = cv2.resize(np.ascontiguousarray(img), (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        if self.train:                                                    # random placement / crop
            ox, oy = random.randint(min(0, self.size - nw), max(0, self.size - nw)), random.randint(min(0, self.size - nh), max(0, self.size - nh))
        else:
            ox, oy = (self.size - nw) // 2, (self.size - nh) // 2
        x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + nw, self.size), min(oy + nh, self.size)
        canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        out = torch.from_numpy(canvas[:, :, ::-1].transpose(2, 0, 1).copy())          # uint8 RGB CHW
        if self.label_dir is None:
            return out
        S, lab = self.size, np.zeros((0, 5), np.float32)
        f = self.label_dir / (Path(self.files[i]).stem + ".txt")
        if f.exists() and f.stat().st_size:                             # YOLO txt: cls xc yc w h, normalised to the original image
            l = np.loadtxt(f, ndmin=2).astype(np.float32)
            if flipped:
                l[:, 1] = 1 - l[:, 1]
            cx, cy, bw, bh = l[:, 1] * nw + ox, l[:, 2] * nh + oy, l[:, 3] * nw, l[:, 4] * nh
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            a0 = np.maximum((x2 - x1) * (y2 - y1), 1e-6)
            x1, x2, y1, y2 = x1.clip(0, S), x2.clip(0, S), y1.clip(0, S), y2.clip(0, S)
            keep = ((x2 - x1) > 2) & ((y2 - y1) > 2) & ((x2 - x1) * (y2 - y1) > 0.25 * a0)
            lab = np.stack([l[:, 0], (x1 + x2) / 2 / S, (y1 + y2) / 2 / S, (x2 - x1) / S, (y2 - y1) / S], 1)[keep]
        return out, torch.from_numpy(lab)


def collate_with_labels(batch):
    imgs = torch.stack([b[0] for b in batch])
    t = [torch.cat([torch.full((len(b[1]), 1), float(i)), b[1]], 1) for i, b in enumerate(batch)]
    return imgs, torch.cat(t, 0)


class EMA:
    """Exponential moving average of the student's weights (as in YOLO training); the EMA model is what gets scored/exported."""

    def __init__(self, model, decay=0.9998, tau=1500):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.updates, self.decay, self.tau = 0, decay, tau

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


class Alpha:
    value = 0.0                                   # shared by every BlendAct: 0 = pure SiLU, 1 = pure ReLU


class BlendAct(torch.nn.Module):
    """(1 - alpha) * SiLU(x) + alpha * ReLU(x). alpha is raised from 0 to 1 during training (progressive replacement)."""

    def forward(self, x):
        a = Alpha.value
        return F.relu(x) if a >= 1.0 else F.silu(x) if a <= 0.0 else (1 - a) * F.silu(x) + a * F.relu(x)


def to_blend(model):
    n = 0
    for i, layer in enumerate(model.model):
        if i >= 24:                                # the Detect head has no activation and is never touched
            continue
        for mod in layer.modules():
            for name, child in list(mod.named_children()):
                if isinstance(child, torch.nn.SiLU):
                    setattr(mod, name, BlendAct()); n += 1
    return n


def select_modules(model, share, max_layer):
    """Names of the SiLU modules to replace: the biggest activations (cost ~ tensor size) in layers <= max_layer,
    taken until they cover `share` of ALL SiLU work."""
    from activation_cost import measure, silu_modules
    sizes = measure(copy.deepcopy(model).float().cpu())
    total = sum(sizes.values())
    layer_of = {n: i for n, i, _ in silu_modules(model)}
    chosen, cum = [], 0
    for n, sz in sorted(sizes.items(), key=lambda kv: -kv[1]):
        if cum / total >= share:
            break
        if layer_of[n] <= max_layer:
            chosen.append(n); cum += sz
    return chosen, cum / total


def to_blend_selected(model, names):
    for full in names:
        i, path = full.split(".", 1)
        layer = model.model[int(i)]
        parent_path, _, child = path.rpartition(".")
        parent = layer.get_submodule(parent_path) if parent_path else layer
        setattr(parent, child, BlendAct())
    return len(names)


def to_relu(model):
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, BlendAct):
                setattr(mod, name, torch.nn.ReLU(inplace=True))


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return (ckpt.get("ema") or ckpt["model"]).float()


class Taps:
    """Capture the outputs of selected layers."""

    def __init__(self, model, layers):
        self.out, self.model, self.layers, self.handles = {}, model, layers, []
        self.attach()

    def attach(self):
        for i in self.layers:
            self.handles.append(self.model.model[i].register_forward_hook(lambda m, inp, o, i=i: self.out.__setitem__(i, o)))

    def remove(self):                       # hooks cannot be pickled, so they are detached while saving
        for h in self.handles:
            h.remove()
        self.handles = []


def run(model, taps, x):
    raw = model(x)                       # Detect.training=True -> raw maps (bs, na, ny, nx, 85) per scale
    return raw, [taps.out[i] for i in FEATURE_LAYERS]


def distill_loss(raw_s, feat_s, raw_t, feat_t):
    lbox = sum(F.smooth_l1_loss(s[..., :4].float(), t[..., :4].float()) for s, t in zip(raw_s, raw_t)) / len(raw_s)
    lcls = sum(F.binary_cross_entropy_with_logits(s[..., 4:].float(), torch.sigmoid(t[..., 4:].float())) for s, t in zip(raw_s, raw_t)) / len(raw_s)
    lfeat = sum(F.mse_loss(a.float(), b.float()) / (b.float().var() + 1e-6) for a, b in zip(feat_s, feat_t)) / len(feat_s)
    return 2.0 * lbox + 5.0 * lcls + 0.5 * lfeat, (lbox.item(), lcls.item(), lfeat.item())


@torch.no_grad()
def holdout_loss(student, teacher, ts, tt, loader, dev):
    student.eval(); student.model[24].training = True
    tot, n = 0.0, 0
    for x in loader:
        x = x.to(dev).float() / 255
        rt, ft = run(teacher, tt, x)
        rs, fs = run(student, ts, x)
        tot += distill_loss(rs, fs, rt, ft)[0].item() * len(x); n += len(x)
    student.train(); student.model[24].training = True
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640, help="training image size (multiple of 32); 512 for the reduced-resolution models")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--student", default=str(DAY2 / "models/relu/relu_untrained.pt"))
    ap.add_argument("--teacher", default=str(DAY2 / "models/fp32/yolov5n.pt"))
    ap.add_argument("--out", default=str(DAY2 / "models/relu/relu_trained.pt"))
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--anneal-frac", type=float, default=0.6, help="fraction of training over which SiLU is blended into ReLU")
    ap.add_argument("--replace-share", type=float, default=1.0, help="replace the biggest SiLU activations covering this share of all SiLU work (1.0 = all)")
    ap.add_argument("--max-layer", type=int, default=23, help="only replace activations in layers up to this index")
    ap.add_argument("--gt-weight", type=float, default=0.0, help="weight of the real YOLO detection loss on ground-truth boxes (0 = distillation only)")
    ap.add_argument("--ema", action="store_true", help="score and export an exponential moving average of the weights")
    ap.add_argument("--resume", action="store_true", help="continue from the per-epoch checkpoint if one exists (training is killed when the session ends)")
    ap.add_argument("--cpus", default="", help="pin training (and its data workers) to these logical CPUs, e.g. 8,9,10,11, so speed benchmarks on other cores stay clean")
    ap.add_argument("--init", default="silu", choices=["silu", "relu"], help="silu: start from the exact SiLU network and blend (recommended)")
    a = ap.parse_args()
    if a.cpus:
        import psutil
        psutil.Process().cpu_affinity([int(c) for c in a.cpus.split(",")])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)

    files = sorted((DAY2 / "data/train2017/images").glob("*.jpg"))
    if a.max_images:
        files = files[:a.max_images]
    train_f, hold_f = files[:-HOLDOUT], files[-HOLDOUT:]
    print(f"{len(train_f)} training images, {len(hold_f)} held-out images (all train2017)", flush=True)
    use_gt = a.gt_weight > 0
    dl = torch.utils.data.DataLoader(Images(train_f, size=a.imgsz, train=True, label_dir=(DAY2 / "data/train2017/labels") if use_gt else None), batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, drop_last=True, persistent_workers=True, collate_fn=collate_with_labels if use_gt else None)
    hl = torch.utils.data.DataLoader(Images(hold_f, size=a.imgsz, train=False), batch_size=a.batch, num_workers=2)

    teacher = load_model(a.teacher).to(dev).eval()
    student = load_model(a.teacher if a.init == "silu" else a.student)
    if a.init == "silu":
        if a.replace_share >= 1.0 and a.max_layer >= 23:
            print("student starts as the exact SiLU network; blended activations:", to_blend(student), flush=True)
        else:
            chosen, got = select_modules(student, a.replace_share, a.max_layer)
            print(f"student starts as the exact SiLU network; replacing {len(chosen)} of 57 activations = {got * 100:.0f}% of all SiLU work "
                  f"(layers {sorted({int(c.split('.')[0]) for c in chosen})})", flush=True)
            to_blend_selected(student, chosen)
            (DAY2 / "results" / f"hybrid_{Path(a.out).stem}_modules.json").write_text(json.dumps({"replaced": chosen, "share": got}, indent=1))
    student = student.to(dev).train()
    for p in teacher.parameters():
        p.requires_grad_(False)
    head_ref = {k: v.clone().cpu() for k, v in load_model(a.teacher).model[24].state_dict().items()}
    for p in student.parameters():
        p.requires_grad_(True)                                         # released checkpoints store parameters as non-trainable
    for p in student.model[24].parameters():
        p.requires_grad_(False)                                        # Detect head frozen
    teacher.model[24].training = True; student.model[24].training = True   # make Detect return raw maps
    ema = EMA(student) if a.ema else None
    compute_gt = None
    if use_gt:
        from utils.loss import ComputeLoss
        if not hasattr(student, "hyp") or not isinstance(student.hyp, dict):
            student.hyp = dict(box=0.05, cls=0.5, obj=1.0, cls_pw=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, label_smoothing=0.0)
        if not hasattr(student, "gr"):
            student.gr = 1.0
        compute_gt = ComputeLoss(student)
    tt, ts = Taps(teacher, FEATURE_LAYERS), Taps(student, FEATURE_LAYERS)
    te = Taps(ema.m, FEATURE_LAYERS) if ema else None

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl); warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / total))))
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    log, best, it, start_ep = [], 1e9, 0, 1
    ckpt_path = Path(a.out).with_name("_resume_" + Path(a.out).stem + ".pt")
    if a.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        student.load_state_dict(ck["student"])
        if ema:
            ema.m.load_state_dict(ck["ema"]); ema.updates = ck["ema_updates"]
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"]); sched.load_state_dict(ck["sched"])
        log, best, it, start_ep = ck["log"], ck["best"], ck["it"], ck["epoch"] + 1
        print(f"RESUMED from epoch {ck['epoch']} (iteration {it})", flush=True)
    else:
        h0 = holdout_loss(student, teacher, ts, tt, hl, dev)
        print(f"held-out distillation loss before training: {h0:.4f}", flush=True)
        log.append({"epoch": 0, "holdout": h0})
    t0 = time.time()
    for ep in range(start_ep, a.epochs + 1):
        student.train(); student.model[24].training = True
        run_loss = np.zeros(4); nb = 0
        for batch in dl:
            Alpha.value = min(1.0, it / max(a.anneal_frac * total, 1)) if a.init == "silu" else 1.0
            x, targets = batch if use_gt else (batch, None)
            x = x.to(dev, non_blocking=True).float() / 255
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                rt, ft = run(teacher, tt, x)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                rs, fs = run(student, ts, x)
            loss, parts = distill_loss(rs, fs, rt, ft)
            if use_gt:
                lgt, _ = compute_gt([r.float() for r in rs], targets.to(dev))
                loss = loss + a.gt_weight * lgt / len(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step(); it += 1
            if ema:
                ema.update(student)
            run_loss += np.array([loss.item(), *parts]); nb += 1
            if it % 100 == 0:
                r = run_loss / nb
                print(f"ep {ep}/{a.epochs} it {it}/{total} loss {r[0]:.3f} (box {r[1]:.3f} cls {r[2]:.3f} feat {r[3]:.3f}) lr {sched.get_last_lr()[0]:.2e} {it * a.batch / (time.time() - t0):.0f} img/s", flush=True)
        if ema:
            h = holdout_loss(ema.m, teacher, te, tt, hl, dev)
            ema.m.eval(); ema.m.model[24].training = True
        else:
            h = holdout_loss(student, teacher, ts, tt, hl, dev)
        log.append({"epoch": ep, "alpha": Alpha.value, "train_loss": (run_loss / max(nb, 1)).tolist(), "holdout": h, "minutes": (time.time() - t0) / 60})
        print(f"== epoch {ep}: alpha(ReLU share) {Alpha.value:.2f}  held-out loss {h:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        (DAY2 / "results" / "finetune_log.json").write_text(json.dumps(log, indent=1))
        if h < best and Alpha.value >= 1.0:            # only a pure-ReLU network can be exported
            best = h
            import copy
            ts.remove()
            if te:
                te.remove()
            export = copy.deepcopy(ema.m if ema else student).eval()
            to_relu(export); export.model[24].training = False
            torch.save({"model": export.half().cpu(), "ema": None, "epoch": ep}, a.out)
            del export
            ts.attach()
            if te:
                te.attach()
        torch.save({"student": student.state_dict(), "ema": ema.m.state_dict() if ema else None, "ema_updates": ema.updates if ema else 0,
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(), "sched": sched.state_dict(), "log": log, "best": best, "it": it, "epoch": ep}, ckpt_path)
    # verify the head is untouched
    final = load_model(a.out).model[24].state_dict()
    same = all(torch.equal(final[k].float(), head_ref[k].float()) for k in head_ref)
    print("Detect head bit-identical to the original:", same, flush=True)
    log.append({"head_identical": bool(same), "best_holdout": best})
    (DAY2 / "results" / "finetune_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
