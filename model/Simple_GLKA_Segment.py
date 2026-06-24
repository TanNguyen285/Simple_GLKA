

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ═════════════════════════════════════════════════════════════════
# 1. GLKA MODULE
# ═════════════════════════════════════════════════════════════════
class GLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.K   = 13

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid(),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, dilation=1),
            nn.BatchNorm2d(dim),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=3, groups=dim, dilation=3),
            nn.BatchNorm2d(dim),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=4, groups=dim, dilation=2),
            nn.BatchNorm2d(dim),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=6, groups=dim, dilation=3),
            nn.BatchNorm2d(dim),
        )
        self.reparam_conv = None

    def forward(self, x):
        global_conv = self.conv0(x)
        anchor      = global_conv * self.se(global_conv)
        if self.reparam_conv is not None:
            branch_main = self.reparam_conv(global_conv)
        else:
            branch_main = (self.branch1(global_conv) + self.branch2(global_conv) +
                           self.branch3(global_conv) + self.branch4(global_conv))
        return anchor * branch_main

    def switch_to_deploy(self):
        if not hasattr(self, "branch1"):
            return
        w1, b1 = self._fuse_bn(self.branch1)
        w2, b2 = self._fuse_bn(self.branch2)
        w3, b3 = self._fuse_bn(self.branch3)
        w4, b4 = self._fuse_bn(self.branch4)
        W_equiv = (self._to_target_k(w1, d=1) + self._to_target_k(w2, d=3) +
                   self._to_target_k(w3, d=2) + self._to_target_k(w4, d=3))
        B_equiv = b1 + b2 + b3 + b4
        self.reparam_conv = nn.Conv2d(
            self.dim, self.dim, self.K,
            padding=self.K // 2, groups=self.dim, bias=True,
        )
        self.reparam_conv.weight.data = W_equiv
        self.reparam_conv.bias.data   = B_equiv
        del self.branch1, self.branch2, self.branch3, self.branch4

    def _fuse_bn(self, block):
        conv = block[0]; bn = block[1]
        std  = (bn.running_var + bn.eps).sqrt()
        t    = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_conv = conv.bias if conv.bias is not None else \
                 torch.zeros(conv.out_channels, device=conv.weight.device)
        return conv.weight * t, bn.bias + (b_conv - bn.running_mean) * bn.weight / std

    def _to_target_k(self, kernel, d):
        c, m, orig_k, _ = kernel.shape
        kd     = (orig_k - 1) * d + 1
        out    = torch.zeros((c, m, self.K, self.K),
                             device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out


# ═════════════════════════════════════════════════════════════════
# 2. BACKBONE COMPONENTS
# ═════════════════════════════════════════════════════════════════
def conv_bn_relu(in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True),
    )


class EfficientBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride,
                 expansion_ratio=2, use_glka=False):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim        = in_channels * expansion_ratio

        self.expand = conv_bn_relu(in_channels, hidden_dim, kernel_size=1)

        if use_glka:
            self.dw   = conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3,
                                     stride=stride, padding=1,
                                     groups=hidden_dim) if stride == 2 else nn.Identity()
            self.glka = GLKA(hidden_dim)
        else:
            self.dw   = conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3,
                                     stride=stride, padding=1, groups=hidden_dim)
            self.glka = nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.dw(out)
        out = self.glka(out)
        out = self.project(out)
        return identity + out if self.use_residual else out


# ═════════════════════════════════════════════════════════════════
# 3. MULTI-SCALE ENCODER
# ═════════════════════════════════════════════════════════════════
class GLKA_Encoder(nn.Module):
    """
    stem  (stride=2): 3   → 32,  H/2
    stage1(stride×2): 32  → 64,  H/4   → low_feat
    stage2(stride×2): 64  → 128, H/8   → mid_feat
    stage3(stride×2): 128 → 256, H/16  → high_feat
    """
    def __init__(self):
        super().__init__()
        self.stem   = conv_bn_relu(3, 32, kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            EfficientBlock(32,  32,  stride=1, expansion_ratio=2, use_glka=False),
            EfficientBlock(32,  64,  stride=2, expansion_ratio=2, use_glka=True),
        )
        self.stage2 = nn.Sequential(
            EfficientBlock(64,  64,  stride=1, expansion_ratio=2, use_glka=True),
            EfficientBlock(64,  128, stride=2, expansion_ratio=2, use_glka=True),
        )
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 128, stride=1, expansion_ratio=2, use_glka=False),
            EfficientBlock(128, 256, stride=2, expansion_ratio=2, use_glka=False),
        )

    def forward(self, x):
        x         = self.stem(x)
        low_feat  = self.stage1(x)         # (B, 64,  H/4,  W/4)
        mid_feat  = self.stage2(low_feat)  # (B, 128, H/8,  W/8)
        high_feat = self.stage3(mid_feat)  # (B, 256, H/16, W/16)
        return low_feat, mid_feat, high_feat

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, GLKA):
                m.switch_to_deploy()


# ═════════════════════════════════════════════════════════════════
# 4. ASPP-LITE
# ═════════════════════════════════════════════════════════════════
def _cbr(in_c, out_c, k=1, p=0, d=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=p, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True),
    )


class ASPPLite(nn.Module):
    """1×1 + dilated 3×3 (r=2,4) + GAP → project"""
    def __init__(self, in_c, out_c):
        super().__init__()
        mid      = out_c // 4
        self.b0  = _cbr(in_c, mid, k=1)
        self.b1  = _cbr(in_c, mid, k=3, p=2, d=2)
        self.b2  = _cbr(in_c, mid, k=3, p=4, d=4)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), _cbr(in_c, mid))
        self.proj= _cbr(mid * 4, out_c)

    def forward(self, x):
        h, w = x.shape[2:]
        gap  = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=False)
        return self.proj(torch.cat([self.b0(x), self.b1(x), self.b2(x), gap], 1))


# ═════════════════════════════════════════════════════════════════
# 5. SEGMENTATION HEAD (FPN-style 3 scale)
# ═════════════════════════════════════════════════════════════════
class LightSegHead(nn.Module):
    """
    high(256,H/16) → ASPP → up×2 → fuse mid(128,H/8)
                           → up×2 → fuse low(64,H/4)
                           → up×4 → classifier
    """
    def __init__(self, num_classes=21, mid_ch=128):
        super().__init__()
        skip_mid = 32
        skip_low = 16

        self.aspp     = ASPPLite(256, mid_ch)
        self.proj_mid = _cbr(128, skip_mid)
        self.proj_low = _cbr(64,  skip_low)

        self.fuse_mid = _cbr(mid_ch + skip_mid, mid_ch, k=3, p=1)
        self.fuse_low = nn.Sequential(
            _cbr(mid_ch + skip_low, mid_ch, k=3, p=1),
            _cbr(mid_ch, mid_ch, k=3, p=1),
        )
        self.classifier = nn.Conv2d(mid_ch, num_classes, 1)

    def forward(self, low_feat, mid_feat, high_feat, target_size):
        x = self.aspp(high_feat)
        x = F.interpolate(x, size=mid_feat.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse_mid(torch.cat([x, self.proj_mid(mid_feat)], 1))

        x = F.interpolate(x, size=low_feat.shape[2:], mode="bilinear", align_corners=False)
        x = self.fuse_low(torch.cat([x, self.proj_low(low_feat)], 1))

        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════
# 6. MODEL TỔNG THỂ
# ═════════════════════════════════════════════════════════════════
class Simple_GLKA_Seg(nn.Module):
    """
    Standalone semantic segmentation — không cần import file ngoài.
    Args:
        num_classes : VOC=21, Cityscapes=19, custom=N
        mid_ch      : decoder channels (default 128)
    Input : (B, 3, H, W),  H,W ≥ 32
    Output: (B, num_classes, H, W)
    """
    def __init__(self, num_classes=21, mid_ch=128):
        super().__init__()
        self.encoder = GLKA_Encoder()
        self.head    = LightSegHead(num_classes=num_classes, mid_ch=mid_ch)

    def forward(self, x):
        target_size        = x.shape[2:]
        low, mid, high     = self.encoder(x)
        return self.head(low, mid, high, target_size)

    def reparameterize(self):
        self.encoder.reparameterize()


# ═════════════════════════════════════════════════════════════════
# 7. LOSS — CE + Dice
# ═════════════════════════════════════════════════════════════════
class SegLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, dice_weight=0.5):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.dice_weight  = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def dice_loss(self, logits, targets):
        probs           = F.softmax(logits, dim=1)
        valid           = (targets != self.ignore_index)
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0
        B, C, H, W = probs.shape
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets_clamped.unsqueeze(1), 1)
        one_hot[(~valid).unsqueeze(1).expand_as(one_hot)] = 0
        inter = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        return (1 - (2 * inter + 1) / (union + 1)).mean()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice_weight * self.dice_loss(logits, targets)


# ═════════════════════════════════════════════════════════════════
# 8. METRICS
# ═════════════════════════════════════════════════════════════════
def mean_iou(preds, targets, num_classes, ignore_index=255):
    ious = []
    for cls in range(num_classes):
        pred_c   = (preds == cls)
        target_c = (targets == cls) & (targets != ignore_index)
        inter    = (pred_c & target_c).sum().item()
        union    = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


# ═════════════════════════════════════════════════════════════════
# 9. KIỂM TRA
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 21

    model = Simple_GLKA_Seg(num_classes=NUM_CLASSES, mid_ch=128).to(DEVICE).eval()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Simple_GLKA_Seg — standalone")
    print(f"Params: {total:.3f} M")

    x = torch.randn(2, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")

    criterion = SegLoss(NUM_CLASSES)
    mask = torch.randint(0, NUM_CLASSES, (2, 224, 224)).to(DEVICE)
    loss = criterion(out, mask)
    print(f"Loss  : {loss.item():.4f}")
    print(f"mIoU  : {mean_iou(out.argmax(1), mask, NUM_CLASSES):.4f}")

    m2 = copy.deepcopy(model); m2.reparameterize(); m2.eval()
    with torch.no_grad():
        out2 = m2(x)
    diff = (out - out2).abs().max().item()
    print(f"Train vs Deploy max diff: {diff:.2e}  {'✓' if diff < 1e-4 else '✗ FAIL'}")