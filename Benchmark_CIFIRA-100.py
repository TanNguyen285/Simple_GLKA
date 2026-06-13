"""
benchmark_imagenette.py
─────────────────────────────────────────────────────────────────────
Benchmark : Simple_GLKA vs MobileNetV2 / EfficientNet-B0 / RepVGG-A0
Dataset   : ImageNette v2-320  (224×224 crop, 10 class, ~1.5 GB)
            — subset của ImageNet-1K, ảnh thật, auto-download
Device    : RTX 4050 6 GB  →  batch=32, AMP ON
Seed      : 42  (locked toàn bộ)
─────────────────────────────────────────────────────────────────────
"""

import os, sys, time, json, csv, argparse, zipfile, random
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ── Import model của bạn ─────────────────────────────────────────
sys.path.insert(0, r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\model")
from Simple_Anphax import Simple_GLKA, GLKA

# ═════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
IMG_SIZE    = 224
BATCH_SIZE  = 32     # 32 an toàn với 6 GB VRAM ở 224×224

DATA_ROOT    = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\data"
SAVE_ROOT    = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\runs\benchmark_imagenette"
HISTORY_FILE = os.path.join(SAVE_ROOT, "benchmark_history.json")

# ImageNette v2 — 320px version (crop xuống 224 khi train)
IMAGENETTE_URL  = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_DIR  = "imagenette2-320"


# ═════════════════════════════════════════════════════════════════
# 1.  SEED — khóa toàn bộ
# ═════════════════════════════════════════════════════════════════
def lock_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)
    print(f"  [seed] Đã khóa seed = {seed}")


# ═════════════════════════════════════════════════════════════════
# 2.  AUTO-DOWNLOAD ImageNette
# ═════════════════════════════════════════════════════════════════
def download_imagenette(root: str) -> str:
    target = os.path.join(root, IMAGENETTE_DIR)
    if os.path.isdir(target):
        print(f"  [data] Đã có tại {target}, bỏ qua download.")
        return target

    os.makedirs(root, exist_ok=True)
    tgz_path = os.path.join(root, "imagenette2-320.tgz")

    if not os.path.exists(tgz_path):
        print(f"  [data] Đang download ImageNette (~1.5 GB) ...")
        with requests.get(IMAGENETTE_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done  = 0
            with open(tgz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        print(f"\r    {pct:.1f}%  "
                              f"({done>>20} MB / {total>>20} MB)", end="", flush=True)
        print(f"\n  [data] Download xong: {tgz_path}")

    print("  [data] Đang giải nén ...")
    import tarfile
    with tarfile.open(tgz_path, "r:gz") as t:
        t.extractall(root)
    print(f"  [data] Giải nén xong: {target}")
    return target


# ═════════════════════════════════════════════════════════════════
# 3.  DATA LOADERS
#     ImageNette label mapping: noisy_imagenette.csv không bắt buộc
#     ImageFolder tự đọc đúng vì thư mục đã đặt tên theo synset
# ═════════════════════════════════════════════════════════════════
# Tên dễ đọc cho 10 class ImageNette
IMAGENETTE_LABELS = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}

# Top-level function — bắt buộc để Windows multiprocessing pickle được
# Hàm lồng trong closure không pickle được trên Windows spawn mode
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders(data_dir: str, batch: int = BATCH_SIZE, seed: int = SEED):
    # ImageNet chuẩn mean/std
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    g = torch.Generator()
    g.manual_seed(seed)

    kw = dict(num_workers=4, pin_memory=True,
              persistent_workers=True,
              worker_init_fn=_seed_worker,
              generator=g)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   val_tf)

    # In label mapping để kiểm tra
    print(f"  [data] train={len(train_ds)}  val={len(val_ds)}")
    print(f"  [data] Classes: "
          + ", ".join(IMAGENETTE_LABELS.get(c, c) for c in train_ds.classes))

    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  **kw),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, **kw))


# ═════════════════════════════════════════════════════════════════
# 4.  MODEL FACTORY
#     224×224: KHÔNG patch stem — giữ nguyên stride gốc
#     Kernel 13×13 của GLKA hoàn toàn hợp lý ở resolution này
# ═════════════════════════════════════════════════════════════════

# ── Simple_GLKA (ours) ───────────────────────────────────────────
def make_simple_glka():
    # Simple_GLKA(num_classes) — stem là conv_bn_relu Sequential stride=2
    # 224×224 input: stride=2 stem → 112, block stride=2 ×3 → 14×14 feature map
    # GLKA kernel 13×13 hoạt động đúng thiết kế ở resolution này
    # KHÔNG patch stem (chỉ cần patch khi dùng CIFAR 32×32)
    base = Simple_GLKA(num_classes=NUM_CLASSES)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = base

        def forward(self, x):
            # Simple_GLKA.forward() trả về (logits, features) — lấy logits
            out = self.m(x)
            return out[0] if isinstance(out, (tuple, list)) else out

        def reparameterize(self):
            """Gọi switch_to_deploy() cho tất cả GLKA block.
            switch_to_deploy() đã có guard nên gọi nhiều lần không crash,
            nhưng Wrapper chỉ expose 1 entry point duy nhất để tránh nhầm.
            """
            for m in self.m.modules():
                if isinstance(m, GLKA):
                    m.switch_to_deploy()

    return Wrapper()


# ── MobileNetV2 ───────────────────────────────────────────────────
def make_mobilenetv2():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
    return m


# ── EfficientNet-B0 ───────────────────────────────────────────────
def make_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    return m


# ── RepVGG-A0 ─────────────────────────────────────────────────────
class RepVGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.stride = stride

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_ch, out_ch, 3, stride, 1,
                                         groups=groups, bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(in_ch)
                                 if (in_ch == out_ch and stride == 1) else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, 1, groups=groups, bias=False),
                nn.BatchNorm2d(out_ch))
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, groups=groups, bias=False),
                nn.BatchNorm2d(out_ch))

        self.nonlinearity = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(x))
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.nonlinearity(out)

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t   = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_c = (conv.bias if conv.bias is not None
               else torch.zeros(conv.out_channels, device=conv.weight.device))
        return conv.weight * t, bn.bias + (b_c - bn.running_mean) * bn.weight / std

    def _pad_1x1_to_3x3(self, k):
        return nn.functional.pad(k, [1, 1, 1, 1])

    def _fuse_identity(self):
        if self.rbr_identity is None:
            return 0, 0
        bn  = self.rbr_identity
        std = (bn.running_var + bn.eps).sqrt()
        t   = bn.weight / std
        ch  = self.in_ch
        k   = torch.zeros(ch, ch // self.groups, 3, 3, device=bn.weight.device)
        for i in range(ch):
            k[i, i % (ch // self.groups), 1, 1] = 1.0
        return k * t.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * t

    def switch_to_deploy(self):
        if self.deploy:
            return
        k3, b3 = self._fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1[0],   self.rbr_1x1[1])
        ki, bi = self._fuse_identity()
        kernel = k3 + self._pad_1x1_to_3x3(k1) + ki
        bias   = b3 + b1 + bi
        self.rbr_reparam = nn.Conv2d(self.in_ch, self.out_ch, 3,
                                     self.stride, 1, groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data   = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity') and self.rbr_identity is not None:
            self.__delattr__('rbr_identity')
        self.deploy = True


class RepVGG_A0(nn.Module):
    def __init__(self, num_classes=10, deploy=False):
        super().__init__()
        a, b = 0.75, 2.5
        ch = [min(48, int(48 * a)),
              int(64  * a),
              int(128 * a),
              int(256 * a),
              int(512 * b)]

        def _stage(ic, oc, n, stride):
            return nn.Sequential(
                RepVGGBlock(ic, oc, stride=stride, deploy=deploy),
                *[RepVGGBlock(oc, oc, stride=1, deploy=deploy)
                  for _ in range(n - 1)])

        # 224×224: stride=2 gốc ở stage0
        self.stage0 = RepVGGBlock(3, ch[0], stride=2, deploy=deploy)
        self.stage1 = _stage(ch[0], ch[1], 2,  stride=2)
        self.stage2 = _stage(ch[1], ch[2], 4,  stride=2)
        self.stage3 = _stage(ch[2], ch[3], 14, stride=2)
        self.stage4 = _stage(ch[3], ch[4], 1,  stride=1)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(ch[4], num_classes)

    def forward(self, x):
        for stage in [self.stage0, self.stage1, self.stage2,
                      self.stage3, self.stage4]:
            x = stage(x)
        return self.fc(torch.flatten(self.gap(x), 1))

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()


def make_repvgg_a0():
    return RepVGG_A0(num_classes=NUM_CLASSES, deploy=False)


# ── Registry — đổi thứ tự ở đây ─────────────────────────────────
MODELS = {
    "Simple_GLKA (ours)": (make_simple_glka,     "Ours"),
    "MobileNetV2":         (make_mobilenetv2,     "Howard et al. 2018"),
    "EfficientNet-B0":     (make_efficientnet_b0, "Tan & Le 2019"),
    "RepVGG-A0":           (make_repvgg_a0,       "Ding et al. 2021"),
}


# ═════════════════════════════════════════════════════════════════
# 5.  METRICS
# ═════════════════════════════════════════════════════════════════
def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_flops(model) -> float:
    try:
        from thop import profile
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        f, _ = profile(model, inputs=(x,), verbose=False)
        return f / 1e9   # GFLOPs
    except Exception:
        return -1.0

def measure_latency(model, n: int = 200) -> float:
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        for _ in range(20): model(x)          # warm-up
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n): model(x)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000   # ms


# ═════════════════════════════════════════════════════════════════
# 6.  TRAIN / EVAL  — AMP + top-5 accuracy
# ═════════════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    top1_correct = top5_correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        if training:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out    = model(imgs)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), torch.cuda.amp.autocast():
                out    = model(imgs)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss   = criterion(logits, labels)

        total_loss   += loss.item() * imgs.size(0)
        top1_correct += logits.argmax(1).eq(labels).sum().item()

        # Top-5 (với 10 class thì top-5 gần 100%, nhưng giữ để đầy đủ)
        _, pred5 = logits.topk(min(5, NUM_CLASSES), dim=1)
        top5_correct += pred5.eq(labels.unsqueeze(1)).any(1).sum().item()
        total        += imgs.size(0)

    n = total
    return (total_loss / n,
            top1_correct / n * 100,
            top5_correct / n * 100)


# ═════════════════════════════════════════════════════════════════
# 7.  TRAIN 1 MODEL
# ═════════════════════════════════════════════════════════════════
def train_one(name, factory, train_loader, val_loader,
              epochs, save_dir, seed=SEED):
    print(f"\n{'═'*68}")
    print(f"  [{name}]")
    print(f"{'═'*68}")

    lock_seed(seed)                          # re-lock trước mỗi model
    model  = factory().to(DEVICE)
    params = count_params(model)
    flops  = count_flops(model)
    print(f"  Params : {params:.3f} M")
    print(f"  GFLOPs : {flops:.3f} G  (input 224×224)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-4)
    scaler    = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    safe_name = (name.replace(" ", "_").replace("/", "-")
                     .replace("(", "").replace(")", ""))
    out_dir   = os.path.join(save_dir, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    # Lưu config để reproduce
    cfg = dict(seed=seed, epochs=epochs, batch=BATCH_SIZE,
               img_size=IMG_SIZE, optimizer="SGD",
               lr=0.1, momentum=0.9, weight_decay=1e-4,
               scheduler="CosineAnnealing", label_smoothing=0.1)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    history = []
    for ep in range(epochs):
        t0 = time.time()
        tr_l, tr1, _   = run_epoch(model, train_loader, criterion,
                                    optimizer, scaler)
        vl_l, vl1, vl5 = run_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        if vl1 > best_acc:
            best_acc = vl1
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "best.pth"))

        history.append({
            "ep": ep + 1,
            "tr_loss": round(tr_l, 4), "vl_loss": round(vl_l, 4),
            "tr_top1": round(tr1,  2),
            "vl_top1": round(vl1,  2), "vl_top5": round(vl5, 2),
        })
        print(f"  Ep {ep+1:3d}/{epochs} | "
              f"TrLoss {tr_l:.4f}  TrTop1 {tr1:.2f}% | "
              f"VlLoss {vl_l:.4f}  VlTop1 {vl1:.2f}%  "
              f"VlTop5 {vl5:.2f}% | "
              f"Best {best_acc:.2f}%  [{elapsed:.1f}s]")

    # ── Load best → reparam → latency ──
    model.load_state_dict(
        torch.load(os.path.join(out_dir, "best.pth"), map_location=DEVICE))
    # Simple_GLKA: dùng Wrapper.reparameterize() — entry point duy nhất,
    #              không gọi thêm model.m.* để tránh double-call
    # RepVGG-A0  : dùng switch_to_deploy() trực tiếp
    if hasattr(model, "reparameterize"):
        model.reparameterize()
    elif hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()

    model.eval()
    _, final_top1, final_top5 = run_epoch(model, val_loader, criterion)
    lat_ms = measure_latency(model)

    result = dict(
        name       = name,
        params_M   = round(params,     3),
        gflops     = round(flops,      3),
        best_top1  = round(best_acc,   2),
        final_top1 = round(final_top1, 2),
        final_top5 = round(final_top5, 2),
        latency_ms = round(lat_ms,     3),
        seed       = seed,
    )
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"summary": result, "history": history}, f, indent=2)

    print(f"\n  ✓ Best={best_acc:.2f}%  "
          f"Final top1={final_top1:.2f}%  top5={final_top5:.2f}%  "
          f"Latency={lat_ms:.3f} ms")
    return result


# ═════════════════════════════════════════════════════════════════
# 8.  HISTORY
# ═════════════════════════════════════════════════════════════════
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(new_results):
    history = load_history()
    entry = {
        "run_id":    len(history) + 1,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device":    DEVICE,
        "dataset":   "ImageNette-v2  224×224  10-class",
        "seed":      SEED,
        "results":   new_results,
    }
    history.append(entry)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n  Lịch sử lưu tại : {HISTORY_FILE}  "
          f"(tổng {len(history)} lần chạy)")

def print_history():
    history = load_history()
    if not history:
        print("  Chưa có lịch sử.")
        return
    print(f"\n{'═'*92}")
    print(f"  LỊCH SỬ BENCHMARK — ImageNette 224×224  ({len(history)} lần chạy)")
    print(f"{'═'*92}")
    for run in history:
        print(f"\n  Run #{run['run_id']}  |  {run['timestamp']}  "
              f"|  {run['device']}  |  seed={run.get('seed', '?')}")
        print(f"  {'Model':<28} {'Params(M)':>9} {'GFLOPs':>8} "
              f"{'Top-1%':>8} {'Top-5%':>8} {'Lat(ms)':>9}")
        print(f"  {'-'*72}")
        for r in sorted(run["results"], key=lambda x: -x["best_top1"]):
            flag = "  ◄ OURS" if "ours" in r["name"].lower() else ""
            print(f"  {r['name']:<28} {r['params_M']:>9.3f} {r['gflops']:>8.3f} "
                  f"{r['best_top1']:>8.2f} {r['final_top5']:>8.2f} "
                  f"{r['latency_ms']:>9.3f}{flag}")
    print(f"\n{'═'*92}")


# ═════════════════════════════════════════════════════════════════
# 9.  BẢNG KẾT QUẢ
# ═════════════════════════════════════════════════════════════════
def print_table(results):
    results = sorted(results, key=lambda x: -x["best_top1"])
    print(f"\n\n{'═'*85}")
    print(f"  BENCHMARK — ImageNette-v2  |  224×224  |  "
          f"From Scratch  |  seed={SEED}  |  {DEVICE}")
    print(f"{'═'*85}")
    print(f"  {'Model':<28} {'Params(M)':>9} {'GFLOPs':>8} "
          f"{'Top-1%':>8} {'Top-5%':>8} {'Lat(ms)':>9}")
    print(f"  {'-'*72}")
    for r in results:
        flag = "  ◄ OURS" if "ours" in r["name"].lower() else ""
        print(f"  {r['name']:<28} {r['params_M']:>9.3f} {r['gflops']:>8.3f} "
              f"{r['best_top1']:>8.2f} {r['final_top5']:>8.2f} "
              f"{r['latency_ms']:>9.3f}{flag}")
    print(f"{'═'*85}")

def save_csv(results, save_dir):
    path   = os.path.join(save_dir, "benchmark_summary.csv")
    fields = ["name", "params_M", "gflops", "best_top1",
              "final_top1", "final_top5", "latency_ms", "seed"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"  CSV lưu tại: {path}")


# ═════════════════════════════════════════════════════════════════
# 10. MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int, default=100,
                        help="Epochs mỗi model (100 đủ hội tụ với ImageNette)")
    parser.add_argument("--batch",       type=int, default=BATCH_SIZE)
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--data",        default=DATA_ROOT)
    parser.add_argument("--output",      default=SAVE_ROOT)
    parser.add_argument("--only",        nargs="+", default=None,
                        help="Chỉ train model chứa chuỗi này")
    parser.add_argument("--skip",        nargs="+", default=None,
                        help="Bỏ qua model chứa chuỗi này")
    parser.add_argument("--history",     action="store_true",
                        help="Chỉ in lịch sử, không train")
    parser.add_argument("--no-download", action="store_true",
                        help="Bỏ qua download, dùng data đã có")
    args = parser.parse_args()

    if args.history:
        print_history()
        return

    lock_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # ── Download ──
    if not args.no_download:
        data_dir = download_imagenette(args.data)
    else:
        data_dir = os.path.join(args.data, IMAGENETTE_DIR)

    print(f"\n  Device  : {DEVICE}")
    print(f"  Seed    : {args.seed}  (locked toàn bộ — reproducible)")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  ImgSize : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Data    : {data_dir}")
    print(f"  Output  : {args.output}")
    print(f"  AMP     : ON")

    train_loader, val_loader = get_loaders(data_dir, args.batch, args.seed)

    # ── Lọc model ──
    selected = dict(MODELS)
    if args.only:
        selected = {k: v for k, v in selected.items()
                    if any(s.lower() in k.lower() for s in args.only)}
    if args.skip:
        selected = {k: v for k, v in selected.items()
                    if not any(s.lower() in k.lower() for s in args.skip)}

    print(f"\n  Model sẽ train: {list(selected.keys())}\n")

    results = []
    for name, (factory, _) in selected.items():
        try:
            r = train_one(name, factory, train_loader, val_loader,
                          args.epochs, args.output, args.seed)
            results.append(r)
        except Exception as e:
            print(f"\n  [!] Lỗi {name}: {e}")
            import traceback; traceback.print_exc()

    if results:
        print_table(results)
        save_csv(results, args.output)
        save_history(results)


if __name__ == "__main__":
    main()