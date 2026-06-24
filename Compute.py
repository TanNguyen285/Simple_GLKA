"""
recompute_metrics.py
─────────────────────────────────────────────────────────────────────
Tính lại Params / GFLOPs / Latency cho cả TRAIN-mode và DEPLOY-mode
của 4 model: Simple_GLKA (ours), MobileNetV2, EfficientNet-B0, RepVGG-A0

Không cần train lại — load best.pth đã có từ benchmark trước.
Nếu thiếu checkpoint của model nào, script vẫn tính được Params/GFLOPs/Latency
(với weight random) cho model đó — chỉ thiếu phần "load đúng weight đã train".
─────────────────────────────────────────────────────────────────────
"""

import os, sys, time, copy
import torch
import torch.nn as nn
from torchvision import models

# ── Import model của bạn ─────────────────────────────────────────
sys.path.insert(0, r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\model")
from Simple_Anphax import Simple_GLKA, GLKA

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 224
NUM_CLASSES = 10

RUN_ROOT = r"C:\Users\ThisPC\Documents\GitHub\CNN_edge\runs\benchmark_imagenette"

CKPT_PATHS = {
    "Simple_GLKA (ours)": os.path.join(RUN_ROOT, "Simple_GLKA_ours",     "best.pth"),
    "MobileNetV2":        os.path.join(RUN_ROOT, "MobileNetV2",         "best.pth"),
    "EfficientNet-B0":    os.path.join(RUN_ROOT, "EfficientNet-B0",     "best.pth"),
    "RepVGG-A0":          os.path.join(RUN_ROOT, "RepVGG-A0",           "best.pth"),
}


# ═════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═════════════════════════════════════════════════════════════════
def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model) -> float:
    try:
        from thop import profile
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        f, _ = profile(model, inputs=(x,), verbose=False)
        return f / 1e9
    except Exception as e:
        print("  [!] thop lỗi:", e)
        return -1.0


def measure_latency(model, n: int = 200) -> float:
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        for _ in range(20):
            model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# ═════════════════════════════════════════════════════════════════
# WRAPPERS — model nào có reparam thì cung cấp reparameterize()
# ═════════════════════════════════════════════════════════════════
class GLKAWrapper(nn.Module):
    """Wrapper cho Simple_GLKA — match đúng key trong checkpoint vì
    benchmark_imagenette.py cũng lưu state_dict từ Wrapper (self.m.*)"""
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


# ── RepVGG-A0 (copy từ benchmark_imagenette.py) ───────────────────
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


# ═════════════════════════════════════════════════════════════════
# REGISTRY — factory + có reparam hay không
# ═════════════════════════════════════════════════════════════════
MODELS = {
    "Simple_GLKA (ours)": lambda: GLKAWrapper(),
    "MobileNetV2":        make_mobilenetv2,
    "EfficientNet-B0":    make_efficientnet_b0,
    "RepVGG-A0":          lambda: RepVGG_A0(num_classes=NUM_CLASSES, deploy=False),
}


def has_reparam(model) -> bool:
    return hasattr(model, "reparameterize") or hasattr(model, "switch_to_deploy")


def do_reparam(model):
    if hasattr(model, "reparameterize"):
        model.reparameterize()
    elif hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()


# ═════════════════════════════════════════════════════════════════
# EVALUATE 1 MODEL
# ═════════════════════════════════════════════════════════════════
def evaluate(name, factory, ckpt_path):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    model = factory().to(DEVICE)

    loaded_ckpt = False
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)
            loaded_ckpt = True
            print(f"  [ckpt] Đã load: {ckpt_path}")
        except Exception as e:
            print(f"  [!] Load checkpoint lỗi ({e}) — dùng weight random.")
    else:
        print(f"  [!] Không tìm thấy checkpoint ({ckpt_path}) — dùng weight random.")

    model.eval()

    # ── TRAIN MODE (multi-branch / pre-reparam) ──
    params_train = count_params(model)
    flops_train  = count_flops(copy.deepcopy(model).to(DEVICE))
    lat_train    = measure_latency(copy.deepcopy(model).to(DEVICE))

    print(f"\n  [Train mode]")
    print(f"    Params : {params_train:.3f} M")
    print(f"    GFLOPs : {flops_train:.3f} G")
    print(f"    Latency: {lat_train:.3f} ms")

    row = dict(
        name=name, loaded_ckpt=loaded_ckpt,
        params_train=round(params_train, 3),
        flops_train=round(flops_train, 3),
        lat_train=round(lat_train, 3),
        params_deploy=None, flops_deploy=None, lat_deploy=None,
        diff=None,
    )

    # ── DEPLOY MODE (reparam, nếu có) ──
    if has_reparam(model):
        model_deploy = copy.deepcopy(model)
        do_reparam(model_deploy)
        model_deploy.eval()

        params_deploy = count_params(model_deploy)
        flops_deploy  = count_flops(model_deploy)
        lat_deploy    = measure_latency(model_deploy)

        print(f"\n  [Deploy mode / reparam]")
        print(f"    Params : {params_deploy:.3f} M")
        print(f"    GFLOPs : {flops_deploy:.3f} G")
        print(f"    Latency: {lat_deploy:.3f} ms")

        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        with torch.no_grad():
            out_train  = model(x)
            out_deploy = model_deploy(x)
        diff = (out_train - out_deploy).abs().max().item()
        print(f"\n  [Sanity] Train vs Deploy max diff: {diff:.2e}  "
              f"{'✓ OK' if diff < 1e-4 else '✗ FAIL — reparam có vấn đề'}")

        row.update(
            params_deploy=round(params_deploy, 3),
            flops_deploy=round(flops_deploy, 3),
            lat_deploy=round(lat_deploy, 3),
            diff=diff,
        )
    else:
        print(f"\n  [Deploy mode] — model không có reparam, dùng số train mode.")

    return row


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    results = []
    for name, factory in MODELS.items():
        ckpt = CKPT_PATHS.get(name)
        try:
            results.append(evaluate(name, factory, ckpt))
        except Exception as e:
            print(f"\n  [!] Lỗi {name}: {e}")
            import traceback; traceback.print_exc()

    # ── Bảng tổng hợp ──
    print(f"\n\n{'='*115}")
    print(f"  TỔNG HỢP — Train mode vs Deploy mode (224×224, {DEVICE})")
    print(f"{'='*115}")
    print(f"  {'Model':<22} {'Ckpt':>5} | {'P_train':>8} {'F_train':>8} {'L_train':>8} | "
          f"{'P_deploy':>9} {'F_deploy':>9} {'L_deploy':>9} | {'diff':>10}")
    print(f"  {'-'*110}")
    for r in results:
        ck = "✓" if r["loaded_ckpt"] else "✗"
        pd = f"{r['params_deploy']:.3f}" if r["params_deploy"] is not None else "—"
        fd = f"{r['flops_deploy']:.3f}"  if r["flops_deploy"]  is not None else "—"
        ld = f"{r['lat_deploy']:.3f}"    if r["lat_deploy"]    is not None else "—"
        df = f"{r['diff']:.2e}"          if r["diff"]          is not None else "—"
        print(f"  {r['name']:<22} {ck:>5} | {r['params_train']:>8.3f} {r['flops_train']:>8.3f} "
              f"{r['lat_train']:>8.3f} | {pd:>9} {fd:>9} {ld:>9} | {df:>10}")
    print(f"{'='*115}")

    print("\n  Lưu ý:")
    print("  - 'Ckpt'=✗  → model dùng weight random (không tìm thấy best.pth),")
    print("    Params/GFLOPs vẫn đúng (kiến trúc cố định), chỉ Latency có thể dao động nhẹ.")
    print("  - 'P/F/L_deploy' = sau khi gọi reparam/switch_to_deploy(); model không")
    print("    có reparam (MobileNetV2, EfficientNet-B0) sẽ để trống ('—').")
    print("  - 'diff' = sai số max |train - deploy| của output logits, phải < 1e-4")
    print("    để khẳng định reparam tương đương về toán học.")


if __name__ == "__main__":
    main()