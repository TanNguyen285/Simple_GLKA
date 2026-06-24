"""
fps_accuracy_table.py
─────────────────────────────────────────────────────────────────────
Bảng so sánh FPS (deploy-mode, batch=1) + Accuracy (Top-1/Top-5)
cho 4 model: Simple_GLKA (ours), MobileNetV2, EfficientNet-B0, RepVGG-A0

- Accuracy : load best.pth đã train, eval lại trên val set ImageNette2-320
- FPS      : batch=1, deploy-mode (sau reparam/switch_to_deploy), FPS = 1000/latency_ms
- Params/GFLOPs : đo ở deploy-mode (số dùng thực tế khi inference)

Không train lại — chỉ cần best.pth + thư mục imagenette2-320/val
─────────────────────────────────────────────────────────────────────
"""

import os, sys, time, copy, csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

sys.path.insert(0, r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\model")
from Simple_Anphax import Simple_GLKA, GLKA

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 224
NUM_CLASSES = 10
BATCH_EVAL  = 32   # batch dùng khi eval accuracy trên val set (không liên quan FPS)

RUN_ROOT  = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\runs\benchmark_imagenette"
DATA_ROOT = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\data"
VAL_DIR   = os.path.join(DATA_ROOT, "imagenette2-320", "val")

OUTPUT_CSV = os.path.join(RUN_ROOT, "fps_accuracy_summary.csv")

CKPT_PATHS = {
    "Simple_GLKA (ours)": os.path.join(RUN_ROOT, "Simple_GLKA_ours",  "best.pth"),
    "MobileNetV2":        os.path.join(RUN_ROOT, "MobileNetV2",       "best.pth"),
    "EfficientNet-B0":    os.path.join(RUN_ROOT, "EfficientNet-B0",   "best.pth"),
    "RepVGG-A0":          os.path.join(RUN_ROOT, "RepVGG-A0",         "best.pth"),
}


# ═════════════════════════════════════════════════════════════════
# MODEL DEFS
# ═════════════════════════════════════════════════════════════════
class GLKAWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = Simple_GLKA(num_classes=NUM_CLASSES)

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out

    def reparameterize(self):
        for m in self.m.modules():
            if isinstance(m, GLKA):
                m.switch_to_deploy()


def make_mobilenetv2():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
    return m


def make_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    return m


# ── RepVGG-A0 ──────────────────────────────────────────────────────
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


MODELS = {
    "Simple_GLKA (ours)": lambda: GLKAWrapper(),
    "MobileNetV2":        make_mobilenetv2,
    "EfficientNet-B0":    make_efficientnet_b0,
    "RepVGG-A0":          lambda: RepVGG_A0(num_classes=NUM_CLASSES, deploy=False),
}


def has_reparam(model):
    return hasattr(model, "reparameterize") or hasattr(model, "switch_to_deploy")


def do_reparam(model):
    if hasattr(model, "reparameterize"):
        model.reparameterize()
    elif hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()


# ═════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model):
    try:
        from thop import profile
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        f, _ = profile(model, inputs=(x,), verbose=False)
        return f / 1e9
    except Exception as e:
        print("  [!] thop lỗi:", e)
        return -1.0


def measure_latency_fps(model, n=500, warmup=100, repeats=7):
    """
    batch=1, trả về (latency_ms, fps).

    - warmup lớn (50) để GPU clock ổn định trước khi đo.
    - đo bằng cuda Event (chính xác hơn perf_counter cho GPU timing).
    - lặp `repeats` lần, mỗi lần n iters, lấy MEDIAN của các lần
      để loại bỏ outlier do jitter hệ thống.
    """
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    run_means = []
    for _ in range(repeats):
        if DEVICE == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            with torch.no_grad():
                for _ in range(n):
                    model(x)
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)  # total ms for n iters
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n):
                    model(x)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        run_means.append(elapsed_ms / n)

    run_means.sort()
    lat_ms = run_means[len(run_means) // 2]  # median
    fps = 1000.0 / lat_ms
    return lat_ms, fps


# ═════════════════════════════════════════════════════════════════
# VAL LOADER
# ═════════════════════════════════════════════════════════════════
def get_val_loader():
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_ds = datasets.ImageFolder(VAL_DIR, val_tf)
    print(f"  [data] val={len(val_ds)}  classes={val_ds.classes}")
    return DataLoader(val_ds, batch_size=BATCH_EVAL, shuffle=False,
                       num_workers=4, pin_memory=True)


@torch.no_grad()
def eval_accuracy(model, loader):
    model.eval()
    top1 = top5 = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        top1 += logits.argmax(1).eq(labels).sum().item()
        _, pred5 = logits.topk(min(5, NUM_CLASSES), dim=1)
        top5 += pred5.eq(labels.unsqueeze(1)).any(1).sum().item()
        total += imgs.size(0)
    return top1 / total * 100, top5 / total * 100


# ═════════════════════════════════════════════════════════════════
# EVALUATE 1 MODEL — deploy mode only
# ═════════════════════════════════════════════════════════════════
def evaluate(name, factory, ckpt_path, val_loader):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    model = factory().to(DEVICE)

    loaded_ckpt = False
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            # Lọc bỏ key thừa do thop.profile gắn vào (total_ops/total_params)
            # khi checkpoint lỡ được lưu sau khi model đã bị thop "đụng" vào.
            state = {k: v for k, v in state.items()
                     if not (k.endswith("total_ops") or k.endswith("total_params"))}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  [!] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  [!] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            loaded_ckpt = (len(missing) == 0)
            if loaded_ckpt:
                print(f"  [ckpt] Đã load: {ckpt_path}")
            else:
                print(f"  [!] Load thiếu key — kiểm tra lại checkpoint/kiến trúc.")
        except Exception as e:
            print(f"  [!] Load checkpoint lỗi ({e}) — dùng weight random.")
    else:
        print(f"  [!] Không tìm thấy checkpoint ({ckpt_path}) — dùng weight random.")

    model.eval()

    # ── Accuracy: eval ở train-mode (trước reparam) để dùng đúng weight đã load ──
    print(f"\n  [Accuracy] Đang eval trên val set...")
    top1, top5 = eval_accuracy(model, val_loader)
    print(f"    Top-1: {top1:.2f}%   Top-5: {top5:.2f}%")

    # ── Reparam → deploy mode ──
    if has_reparam(model):
        do_reparam(model)
        model.eval()

        # sanity check: accuracy sau reparam phải gần như y nguyên
        top1_dep, top5_dep = eval_accuracy(model, val_loader)
        print(f"  [Sanity] Top-1 sau reparam: {top1_dep:.2f}%  "
              f"(diff={abs(top1-top1_dep):.4f}%)")
    else:
        print(f"  [Deploy] Model không có reparam, dùng nguyên trạng.")

    # ── Params / GFLOPs / FPS ở deploy-mode ──
    params = count_params(model)
    flops  = count_flops(copy.deepcopy(model))
    lat_ms, fps = measure_latency_fps(model)

    print(f"\n  [Deploy mode]")
    print(f"    Params : {params:.3f} M")
    print(f"    GFLOPs : {flops:.3f} G")
    print(f"    Latency: {lat_ms:.3f} ms")
    print(f"    FPS    : {fps:.1f}")

    return dict(
        name=name, loaded_ckpt=loaded_ckpt,
        params_M=round(params, 3), gflops=round(flops, 3),
        top1=round(top1, 2), top5=round(top5, 2),
        latency_ms=round(lat_ms, 3), fps=round(fps, 1),
    )


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    if not os.path.isdir(VAL_DIR):
        print(f"  [!] Không tìm thấy val dir: {VAL_DIR}")
        print(f"      Sửa DATA_ROOT trong script cho đúng đường dẫn imagenette2-320.")
        return

    # cudnn benchmark=True giúp chọn algorithm tối ưu và ổn định hơn
    # sau vài lần warm-up (quan trọng với input cố định 1x3x224x224)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    # Warm-up GPU clock chung trước khi đo model nào — tránh model
    # đầu tiên bị "lạnh" GPU còn model sau hưởng lợi clock boost
    if DEVICE == "cuda":
        print("  [warmup] Đang làm nóng GPU...")
        dummy = nn.Conv2d(3, 64, 3, padding=1).to(DEVICE)
        x = torch.randn(8, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            for _ in range(100):
                dummy(x)
        torch.cuda.synchronize()

    val_loader = get_val_loader()

    results = []
    for name, factory in MODELS.items():
        ckpt = CKPT_PATHS.get(name)
        try:
            results.append(evaluate(name, factory, ckpt, val_loader))
        except Exception as e:
            print(f"\n  [!] Lỗi {name}: {e}")
            import traceback; traceback.print_exc()

    # ── Bảng tổng hợp ──
    results_sorted = sorted(results, key=lambda x: -x["top1"])

    print(f"\n\n{'='*100}")
    print(f"  BẢNG TỔNG HỢP — Deploy mode, batch=1, {DEVICE}")
    print(f"  Accuracy: eval trên ImageNette2-320 val set ({VAL_DIR})")
    print(f"{'='*100}")
    print(f"  {'Model':<22} {'Ckpt':>5} {'Params(M)':>10} {'GFLOPs':>8} "
          f"{'Top-1%':>8} {'Top-5%':>8} {'Lat(ms)':>9} {'FPS':>8}")
    print(f"  {'-'*92}")
    for r in results_sorted:
        ck = "✓" if r["loaded_ckpt"] else "✗"
        flag = "  ◄ OURS" if "ours" in r["name"].lower() else ""
        print(f"  {r['name']:<22} {ck:>5} {r['params_M']:>10.3f} {r['gflops']:>8.3f} "
              f"{r['top1']:>8.2f} {r['top5']:>8.2f} {r['latency_ms']:>9.3f} "
              f"{r['fps']:>8.1f}{flag}")
    print(f"{'='*100}")

    if any(not r["loaded_ckpt"] for r in results):
        print("\n  [!] Cảnh báo: một số model dùng weight RANDOM (không tìm thấy best.pth)")
        print("      → Accuracy của các model đó KHÔNG có ý nghĩa, chỉ Params/GFLOPs/FPS đáng tin.")

    # ── CSV ──
    os.makedirs(RUN_ROOT, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["name", "loaded_ckpt", "params_M", "gflops",
                  "top1", "top5", "latency_ms", "fps"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results_sorted:
            w.writerow({k: r[k] for k in fields})
    print(f"\n  CSV lưu tại: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()