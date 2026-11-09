"""Train the plate DETECTOR (YOLOX-tiny, 1 class = plate) from the COCO-pretrained YOLOX-tiny.

No govtech weights anywhere: the starting point is Megvii's COCO yolox_tiny.pth (Apache-2.0), with the
classification convs replaced by fresh 1-class ones. Model and loss (SimOTA, IoU/obj/cls/L1) are
Megvii's own code from the YOLOX repository (put its source dir on --yolox-src); the data pipeline is
here, kept to what this task needs:

  * mosaic (4 images on a 2S canvas, then a random affine back to S) for half the samples, a single
    image at a random scale and offset for the rest -- the plates in Open Images and the synthetic
    frames span 1%..60% of the frame, and mosaic is what puts the small end into every batch
  * HSV jitter and horizontal flip
  * the last --no-aug-epochs epochs: no mosaic, plain top-left letterbox (exactly what inference
    does), and the L1 box loss on, as Megvii does

Input convention is the one lpr_cpp's inference uses (wasm/yolox_wasm.cpp, pure/m_plate_detect.cpp):
BGR, 0..255 float, letterboxed into 416x416 with pad 114 at the top-left.

Data dirs are YOLO format (images/*.jpg + labels/*.txt with `cls cx cy w h`, normalised); every
box is taken as a plate whatever its class id.

  python train_plate_yolox.py --yolox-src YOLOX-0.3.0 --init yolox_tiny.pth \
      --data data/ydet_train --data data/det_oi --data data/det_real:8 \
      --val data/det_oi_val --val data/det_real_val --val data/ydet_val --out runs/plate
"""
import argparse
import glob
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

S = 416


# ----------------------------------------------------------------------------------------- data
def read_dir(d):
    items = []
    for ip in sorted(glob.glob(os.path.join(d, "images", "*.jpg")) + glob.glob(os.path.join(d, "images", "*.png"))):
        lp = os.path.join(d, "labels", os.path.splitext(os.path.basename(ip))[0] + ".txt")
        boxes = []
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) >= 5:
                    boxes.append([float(v) for v in p[1:5]])
        items.append((ip, np.array(boxes, np.float32).reshape(-1, 4)))
    return items


def load(item):
    ip, nb = item
    img = cv2.imread(ip, cv2.IMREAD_COLOR)                   # BGR
    h, w = img.shape[:2]
    b = nb.copy()
    xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1) if len(b) else np.zeros((0, 4), np.float32)
    return img, xyxy


def place(canvas, img, xyxy, s, ox, oy):
    """Resize img by s and paste with its top-left at (ox, oy) (may be partly outside). Returns boxes."""
    h, w = img.shape[:2]
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + nw, W), min(oy + nh, H)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = r[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return xyxy * s + np.array([ox, oy, ox, oy], np.float32)


def clip_boxes(b, W, H, min_px=3, keep_frac=0.45):
    if not len(b):
        return b
    area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, W)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, H)
    w, h = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    ok = (w >= min_px) & (h >= min_px) & (w * h >= keep_frac * np.maximum(area0, 1e-6))
    return c[ok]


def hsv_jitter(img, hg=0.015, sg=0.6, vg=0.4):
    r = np.random.uniform(-1, 1, 3) * [hg, sg, vg] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class PlateSet(torch.utils.data.Dataset):
    def __init__(self, items, aug=True, mosaic_p=0.5, length=None):
        self.items, self.aug, self.mosaic_p = items, aug, mosaic_p
        self.length = length or len(items)

    def __len__(self):
        return self.length

    def letterbox(self, img, xyxy):
        canvas = np.full((S, S, 3), 114, np.uint8)
        s = S / max(img.shape[:2])
        return canvas, place(canvas, img, xyxy, s, 0, 0)

    def single(self, img, xyxy):
        canvas = np.full((S, S, 3), 114, np.uint8)
        s = S / max(img.shape[:2]) * random.uniform(0.6, 1.5)
        nw, nh = img.shape[1] * s, img.shape[0] * s
        ox = int(random.uniform(min(0, S - nw), max(0, S - nw)))
        oy = int(random.uniform(min(0, S - nh), max(0, S - nh)))
        return canvas, place(canvas, img, xyxy, s, ox, oy)

    def mosaic(self, idx):
        big = np.full((2 * S, 2 * S, 3), 114, np.uint8)
        xc, yc = int(random.uniform(0.5 * S, 1.5 * S)), int(random.uniform(0.5 * S, 1.5 * S))
        boxes = []
        for k, i in enumerate([idx] + [random.randrange(len(self.items)) for _ in range(3)]):
            img, b = load(self.items[i])
            s = S / max(img.shape[:2])
            nw, nh = int(img.shape[1] * s), int(img.shape[0] * s)
            ox = xc - nw if k in (0, 2) else xc
            oy = yc - nh if k in (0, 1) else yc
            boxes.append(place(big, img, b, s, ox, oy))
        boxes = clip_boxes(np.concatenate(boxes), 2 * S, 2 * S)
        # random affine 2S -> S: scale around the canvas centre, small translation
        sc = random.uniform(0.5, 1.5) * 0.5                    # 0.5 maps the 2S canvas onto S
        tx = random.uniform(-0.1, 0.1) * S
        ty = random.uniform(-0.1, 0.1) * S
        M = np.array([[sc, 0, S / 2 - sc * S + tx], [0, sc, S / 2 - sc * S + ty]], np.float32)
        out = cv2.warpAffine(big, M, (S, S), borderValue=(114, 114, 114))
        b = boxes * sc + np.array([M[0, 2], M[1, 2], M[0, 2], M[1, 2]], np.float32)
        return out, b

    def __getitem__(self, i):
        i = i % len(self.items)
        if self.aug and random.random() < self.mosaic_p:
            img, b = self.mosaic(i)
        else:
            img0, b0 = load(self.items[i])
            img, b = (self.single if self.aug else self.letterbox)(img0, b0)
        b = clip_boxes(b, S, S)
        if self.aug:
            img = hsv_jitter(img)
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                if len(b):
                    b = np.stack([S - b[:, 2], b[:, 1], S - b[:, 0], b[:, 3]], 1)
        lab = np.zeros((120, 5), np.float32)                  # cls, cx, cy, w, h (pixels)
        n = min(len(b), 120)
        if n:
            lab[:n, 1] = (b[:n, 0] + b[:n, 2]) / 2
            lab[:n, 2] = (b[:n, 1] + b[:n, 3]) / 2
            lab[:n, 3] = b[:n, 2] - b[:n, 0]
            lab[:n, 4] = b[:n, 3] - b[:n, 1]
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        return x, torch.from_numpy(lab)


# ----------------------------------------------------------------------------------------- model
def build_model(yolox_src, init):
    sys.path.insert(0, yolox_src)
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    depth, width = 0.33, 0.375                                # yolox_tiny
    backbone = YOLOPAFPN(depth, width, in_channels=[256, 512, 1024], act="silu")
    head = YOLOXHead(80, width, in_channels=[256, 512, 1024], act="silu")
    model = YOLOX(backbone, head)
    if init:
        ck = torch.load(init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model", ck), strict=True)
    # 1 class: fresh classification convs (the COCO ones have 80 outputs)
    for i, c in enumerate(head.cls_preds):
        head.cls_preds[i] = nn.Conv2d(c.in_channels, 1, 1, 1, 0)
    head.num_classes = 1
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    head.initialize_biases(1e-2)
    return model


# ----------------------------------------------------------------------------------------- eval
def ap50(dets_per_img, gts_per_img):
    """VOC-style AP at IoU 0.5 for one class, all-point interpolation; also recall at conf>=0.25."""
    scores, tps, npos, rec25 = [], [], 0, [0, 0]
    for dets, gts in zip(dets_per_img, gts_per_img):
        npos += len(gts)
        used = np.zeros(len(gts), bool)
        for d in sorted(dets, key=lambda d: -d[4]):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                ix = max(0, min(d[2], g[2]) - max(d[0], g[0])); iy = max(0, min(d[3], g[3]) - max(d[1], g[1]))
                inter = ix * iy
                u = (d[2] - d[0]) * (d[3] - d[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
                iou = inter / u if u > 0 else 0
                if iou > best:
                    best, bj = iou, j
            tp = best >= 0.5 and not used[bj]
            if tp:
                used[bj] = True
            scores.append(d[4]); tps.append(tp)
        hit = np.zeros(len(gts), bool)
        for d in dets:
            if d[4] < 0.25:
                continue
            for j, g in enumerate(gts):
                ix = max(0, min(d[2], g[2]) - max(d[0], g[0])); iy = max(0, min(d[3], g[3]) - max(d[1], g[1]))
                inter = ix * iy
                u = (d[2] - d[0]) * (d[3] - d[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
                if u > 0 and inter / u >= 0.5:
                    hit[j] = True
        rec25[0] += hit.sum(); rec25[1] += len(gts)
    if npos == 0:
        return 0.0, 0.0
    o = np.argsort(-np.array(scores))
    tp = np.array(tps, float)[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / npos, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), rec25[0] / max(rec25[1], 1)


@torch.no_grad()
def evaluate(model, items, dev, bs=32):
    from yolox.utils import postprocess
    model.eval()
    ds = PlateSet(items, aug=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    dets, gts = [], []
    for x, lab in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x.to(dev))
        res = postprocess(out.float(), 1, conf_thre=0.001, nms_thre=0.65)
        for r, l in zip(res, lab.numpy()):
            l = l[l[:, 3] > 0]
            gts.append(np.stack([l[:, 1] - l[:, 3] / 2, l[:, 2] - l[:, 4] / 2, l[:, 1] + l[:, 3] / 2, l[:, 2] + l[:, 4] / 2], 1) if len(l) else np.zeros((0, 4)))
            if r is None:
                dets.append([])
            else:
                r = r.cpu().numpy()
                dets.append([[a[0], a[1], a[2], a[3], a[4] * a[5]] for a in r])
    return ap50(dets, gts)


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolox-src", required=True)
    ap.add_argument("--init", default="", help="COCO yolox_tiny.pth")
    ap.add_argument("--data", action="append", required=True, help="dir[:repeat]")
    ap.add_argument("--val", action="append", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-aug-epochs", type=int, default=8)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01, help="per 64 images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-val", type=int, default=1500)
    ap.add_argument("--out", default="runs/plate")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")

    items = []
    for spec in a.data:
        d, rep = (spec.rsplit(":", 1) + ["1"])[:2] if ":" in spec[2:] else (spec, "1")
        it = read_dir(d)
        print(f"train {d}: {len(it)} images x{rep}", flush=True)
        items += it * int(rep)
    vals = {}
    for d in a.val:
        it = read_dir(d)
        if len(it) > a.max_val:
            it = random.Random(0).sample(it, a.max_val)
        vals[os.path.basename(os.path.normpath(d))] = it
        print(f"val {d}: {len(it)} images", flush=True)

    model = build_model(a.yolox_src, a.init).to(dev)
    from yolox.utils import ModelEMA
    pg_bn, pg_w, pg_b = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg_bn.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg_w.append(v.weight)
    lr = a.lr * a.batch / 64
    opt = torch.optim.SGD(pg_bn, lr=lr, momentum=0.9, nesterov=True)
    opt.add_param_group({"params": pg_w, "weight_decay": 5e-4})
    opt.add_param_group({"params": pg_b})
    ema = ModelEMA(model, 0.9998)
    scaler = torch.amp.GradScaler("cuda")

    ds = PlateSet(items, aug=True)
    log = open(os.path.join(a.out, "log.csv"), "a")
    best = -1.0
    it_per_epoch = math.ceil(len(ds) / a.batch)
    total = a.epochs * it_per_epoch
    warm = a.warmup_epochs * it_per_epoch
    step = 0
    model.train()
    for ep in range(a.epochs):
        no_aug = ep >= a.epochs - a.no_aug_epochs
        if no_aug:
            ds.mosaic_p = 0.0
            model.head.use_l1 = True
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                         drop_last=True, persistent_workers=False, pin_memory=True)
        t0, agg, n = time.time(), {}, 0
        for x, lab in dl:
            if step < warm:
                cur = lr * (step / warm) ** 2
            elif no_aug:
                cur = lr * 0.05
            else:
                p = (step - warm) / max(1, total - warm - a.no_aug_epochs * it_per_epoch)
                cur = lr * 0.05 + 0.5 * (lr - lr * 0.05) * (1 + math.cos(math.pi * min(p, 1.0)))
            for g in opt.param_groups:
                g["lr"] = cur
            x, lab = x.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x, lab)
            loss = out["total_loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1
            n += 1
            for k, v in out.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        msg = {k: round(v / max(n, 1), 4) for k, v in agg.items()}
        line = f"epoch {ep + 1}/{a.epochs} lr {cur:.5f} {msg} {time.time() - t0:.0f}s"
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            res = {k: evaluate(ema.ema, v, dev) for k, v in vals.items()}
            real_world = [res[k][0] for k in res if "oi" in k or "real" in k]
            score = float(np.mean(real_world)) if real_world else float(np.mean([r[0] for r in res.values()]))
            line += " | " + " ".join(f"{k} AP50 {r[0]:.3f} R@.25 {r[1]:.3f}" for k, r in res.items())
            torch.save({"model": ema.ema.state_dict(), "epoch": ep + 1, "val": res}, os.path.join(a.out, "last.pth"))
            if score > best:
                best = score
                torch.save({"model": ema.ema.state_dict(), "epoch": ep + 1, "val": res}, os.path.join(a.out, "best.pth"))
                line += f" * best {score:.3f}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()


if __name__ == "__main__":
    main()
