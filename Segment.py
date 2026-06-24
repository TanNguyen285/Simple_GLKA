"""
train_seg.py  — VOC2012 standard protocol
─────────────────────────────────────────────────────────────────────
Chuẩn theo paper (ENet / BiSeNet / DeepLabV3 lite trên VOC1464):
  - img_size  : 512×512
  - optimizer : SGD + momentum=0.9 + weight_decay=1e-4
  - LR        : poly decay (1 - iter/max_iter)^0.9
  - augment   : random scale [0.5,2.0] + random crop 512 + hflip
  - loss      : CrossEntropy (ignore_index=255)
  - epochs    : 60 (≈ paper 50-100)
─────────────────────────────────────────────────────────────────────
Usage:
    python train_seg.py
    python train_seg.py --epochs 80 --bs 8
    python train_seg.py --smoke
    python train_seg.py --resume ./checkpoints/latest.pth
"""

import argparse
import os
import time
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.datasets import VOCSegmentation
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\model")
from Simple_GLKA_Segment import Simple_GLKA_Seg, mean_iou


# ═════════════════════════════════════════════════════════════════════
# CONFIG — chuẩn paper
# ═════════════════════════════════════════════════════════════════════
CFG = dict(
    data_root     = "./data/VOC",
    crop_size     = 512,          # paper chuẩn
    num_classes   = 21,
    ignore_index  = 255,
    batch_size    = 8,            # 6GB VRAM với AMP
    num_workers   = 2,
    epochs        = 60,
    base_lr       = 1e-2,         # SGD thường dùng lr cao hơn AdamW
    momentum      = 0.9,
    weight_decay  = 1e-4,
    poly_power    = 0.9,          # poly LR decay exponent
    mid_ch        = 128,
    save_dir      = "./checkpoints",
    log_interval  = 20,
    # augment scale range
    scale_min     = 0.5,
    scale_max     = 2.0,
)


# ═════════════════════════════════════════════════════════════════════
# DATASET — standard VOC augmentation protocol
# ═════════════════════════════════════════════════════════════════════
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class VOCSegDataset(torch.utils.data.Dataset):
    """
    Train augmentation (chuẩn paper):
      1. Random scale ảnh [0.5, 2.0] × crop_size
      2. Pad nếu ảnh nhỏ hơn crop_size (pad ảnh=mean, pad mask=255)
      3. Random crop crop_size × crop_size
      4. Random horizontal flip 50%
      5. Normalize

    Val:
      1. Resize cạnh ngắn = crop_size (giữ aspect ratio)
      2. Center crop crop_size × crop_size
      3. Normalize
    """
    def __init__(self, root, image_set="train", crop_size=512, download=True):
        self.base      = VOCSegmentation(root=root, year="2012",
                                         image_set=image_set, download=download)
        self.crop_size = crop_size
        self.is_train  = (image_set == "train")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]   # PIL RGB, PIL P
        if self.is_train:
            img, mask = self._train_transform(img, mask)
        else:
            img, mask = self._val_transform(img, mask)
        return img, mask

    # ── train ──────────────────────────────────────────────────────
    def _train_transform(self, img, mask):
        # 1. random scale
        scale = random.uniform(CFG["scale_min"], CFG["scale_max"])
        new_size = int(self.crop_size * scale)
        img  = TF.resize(img,  [new_size, new_size],
                         interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_size, new_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        # 2. pad nếu nhỏ hơn crop_size
        img, mask = self._pad_if_needed(img, mask)

        # 3. random crop
        i, j, h, w = torch.nn.modules.utils._pair(self.crop_size), None, None, None
        i, j, h, w = self._random_crop_params(img)
        img  = TF.crop(img,  i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # 4. random flip
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # 5. to tensor + normalize
        return self._to_tensor(img, mask)

    # ── val ────────────────────────────────────────────────────────
    def _val_transform(self, img, mask):
        # resize cạnh ngắn = crop_size
        w, h    = img.size
        short   = min(w, h)
        scale   = self.crop_size / short
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        img  = TF.resize(img,  [new_h, new_w],
                         interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_h, new_w],
                         interpolation=TF.InterpolationMode.NEAREST)

        # pad nếu cần
        img, mask = self._pad_if_needed(img, mask)

        # center crop
        img  = TF.center_crop(img,  self.crop_size)
        mask = TF.center_crop(mask, self.crop_size)

        return self._to_tensor(img, mask)

    # ── helpers ────────────────────────────────────────────────────
    def _pad_if_needed(self, img, mask):
        w, h = img.size
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            # padding: left, top, right, bottom
            pad = [pad_w // 2, pad_h // 2,
                   pad_w - pad_w // 2, pad_h - pad_h // 2]
            # img: pad với mean value (ImageNet mean × 255)
            fill_img  = tuple(int(m * 255) for m in MEAN)
            img  = TF.pad(img,  pad, fill=fill_img)
            mask = TF.pad(mask, pad, fill=255)   # 255 = ignore
        return img, mask

    def _random_crop_params(self, img):
        w, h = img.size
        th = tw = self.crop_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def _to_tensor(self, img, mask):
        img  = TF.to_tensor(img)
        img  = TF.normalize(img, mean=MEAN, std=STD)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask


def build_loaders(cfg):
    train_ds = VOCSegDataset(cfg["data_root"], "train", cfg["crop_size"])
    val_ds   = VOCSegDataset(cfg["data_root"], "val",   cfg["crop_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=True)
    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


# ═════════════════════════════════════════════════════════════════════
# LOSS — CrossEntropy only (chuẩn paper VOC)
# ═════════════════════════════════════════════════════════════════════
class SegLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


# ═════════════════════════════════════════════════════════════════════
# POLY LR — chuẩn paper segmentation
# ═════════════════════════════════════════════════════════════════════
class PolyLR:
    """
    lr = base_lr × (1 - iter / max_iter) ^ power
    Step mỗi iteration (không phải mỗi epoch).
    """
    def __init__(self, optimizer, max_iters, base_lr, power=0.9, min_lr=1e-6):
        self.optimizer  = optimizer
        self.max_iters  = max_iters
        self.base_lr    = base_lr
        self.power      = power
        self.min_lr     = min_lr
        self.cur_iter   = 0

    def step(self):
        lr = self.base_lr * (1 - self.cur_iter / self.max_iters) ** self.power
        lr = max(lr, self.min_lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.cur_iter += 1
        return lr

    def state_dict(self):
        return {"cur_iter": self.cur_iter}

    def load_state_dict(self, d):
        self.cur_iter = d["cur_iter"]


# ═════════════════════════════════════════════════════════════════════
# TRAIN ONE EPOCH
# ═════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                device, cfg, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for i, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        lr = scheduler.step()   # poly: update mỗi iter
        total_loss += loss.item()

        if (i + 1) % cfg["log_interval"] == 0:
            print(f"  Epoch {epoch:3d} | step {i+1:4d}/{len(loader)} "
                  f"| loss {loss.item():.4f} | lr {lr:.2e} "
                  f"| {time.time()-t0:.1f}s")
            t0 = time.time()

    return total_loss / len(loader)


# ═════════════════════════════════════════════════════════════════════
# VALIDATE
# ═════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    all_miou   = []

    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_miou.append(mean_iou(preds, masks, cfg["num_classes"], cfg["ignore_index"]))

    return total_loss / len(loader), float(np.mean(all_miou))


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=CFG["epochs"])
    parser.add_argument("--bs",     type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",     type=float, default=CFG["base_lr"])
    parser.add_argument("--smoke",  action="store_true")
    parser.add_argument("--resume", type=str,   default=None)
    args = parser.parse_args()

    CFG["epochs"]     = args.epochs
    CFG["batch_size"] = args.bs
    CFG["base_lr"]    = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    os.makedirs(CFG["save_dir"], exist_ok=True)

    train_loader, val_loader = build_loaders(CFG)

    model = Simple_GLKA_Seg(
        num_classes=CFG["num_classes"],
        mid_ch=CFG["mid_ch"],
    ).to(device)
    print(f"[Model] Params: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")

    criterion = SegLoss(ignore_index=CFG["ignore_index"])

    # SGD — chuẩn paper segmentation
    optimizer = optim.SGD(
        model.parameters(),
        lr=CFG["base_lr"],
        momentum=CFG["momentum"],
        weight_decay=CFG["weight_decay"],
        nesterov=True,
    )

    max_iters = CFG["epochs"] * len(train_loader)
    scheduler = PolyLR(optimizer, max_iters=max_iters,
                       base_lr=CFG["base_lr"], power=CFG["poly_power"])
    scaler    = GradScaler()

    # ── Resume ──
    start_epoch = 1
    best_miou   = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"[Resume] Epoch {start_epoch}, best mIoU {best_miou:.4f}")

    # ── Smoke test ──
    if args.smoke:
        print("[Smoke] 1 batch test...")
        imgs, masks = next(iter(train_loader))
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)
        print(f"  Input : {tuple(imgs.shape)}")
        print(f"  Output: {tuple(logits.shape)}")
        print(f"  Loss  : {loss.item():.4f}")
        print("[Smoke] OK ✓")
        return

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"  Training {CFG['epochs']} epochs | crop {CFG['crop_size']}px | "
          f"bs {CFG['batch_size']} | SGD poly lr {CFG['base_lr']} | AMP ON")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, CFG, epoch)
        val_loss, val_miou = validate(
            model, val_loader, criterion, device, CFG)

        is_best = val_miou > best_miou
        print(f"Epoch {epoch:3d}/{CFG['epochs']} "
              f"| train {train_loss:.4f} "
              f"| val {val_loss:.4f} "
              f"| mIoU {val_miou:.4f}"
              + (" ← best" if is_best else ""))

        ckpt = {
            "epoch"    : epoch,
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler"   : scaler.state_dict(),
            "best_miou": best_miou,
        }
        torch.save(ckpt, os.path.join(CFG["save_dir"], "latest.pth"))
        if is_best:
            best_miou = val_miou
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, os.path.join(CFG["save_dir"], "best.pth"))

    print(f"\n[Done] Best mIoU: {best_miou:.4f}")

    # ── Export deploy ──
    print("\n[Export] Reparameterizing...")
    deploy_model = copy.deepcopy(model)
    deploy_model.reparameterize()
    deploy_model.eval()
    torch.save(deploy_model.state_dict(),
               os.path.join(CFG["save_dir"], "deploy.pth"))
    print("[Export] deploy.pth saved ✓")


if __name__ == "__main__":
    main()