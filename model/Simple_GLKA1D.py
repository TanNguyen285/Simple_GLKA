"""
Simple_Anphax_1D.py
─────────────────────────────────────────────────────────────────────
Bản 1D của Simple_GLKA — dùng cho audio waveform / time-series.
Giữ nguyên toàn bộ triết lý GLKA (multi-branch depthwise + SE + reparam),
chỉ thay Conv2d → Conv1d, AdaptiveAvgPool2d → AdaptiveAvgPool1d, v.v.

Input : (B, 1, T) — waveform thô hoặc feature 1D bất kỳ
Output: (logits, features)
─────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════
# 1. GLKA 1D
# ═════════════════════════════════════════════════════════════════
class GLKA1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.K   = 13  # target kernel size sau reparam

        self.conv0 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, max(dim // 8, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(dim // 8, 1), dim, 1),
            nn.Sigmoid()
        )

        # 4 branch: (kernel, dilation) → effective receptive field khác nhau
        self.branch1 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1,  groups=dim, dilation=1),
            nn.BatchNorm1d(dim)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=3,  groups=dim, dilation=3),
            nn.BatchNorm1d(dim)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=4,  groups=dim, dilation=2),
            nn.BatchNorm1d(dim)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=6,  groups=dim, dilation=3),
            nn.BatchNorm1d(dim)
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

        self.reparam_conv = nn.Conv1d(
            self.dim, self.dim, self.K,
            padding=self.K // 2,
            groups=self.dim,
            bias=True
        )
        self.reparam_conv.weight.data = W_equiv
        self.reparam_conv.bias.data   = B_equiv

        del self.branch1, self.branch2, self.branch3, self.branch4

    def _fuse_bn(self, block):
        conv = block[0]
        bn   = block[1]
        std  = (bn.running_var + bn.eps).sqrt()
        t    = (bn.weight / std).reshape(-1, 1, 1)   # (C,1,1) cho 1D
        b_conv  = (conv.bias if conv.bias is not None
                   else torch.zeros(conv.out_channels, device=conv.weight.device))
        w_fused = conv.weight * t
        b_fused = bn.bias + (b_conv - bn.running_mean) * bn.weight / std
        return w_fused, b_fused

    def _to_target_k(self, kernel, d):
        """Đặt kernel nhỏ (với dilation d) vào tensor K chiều, căn giữa."""
        c, m, orig_k = kernel.shape
        kd     = (orig_k - 1) * d + 1          # effective length
        out    = torch.zeros((c, m, self.K), device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset : offset + kd : d] = kernel
        return out


# ═════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═════════════════════════════════════════════════════════════════
def conv_bn_relu_1d(in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding,
                  groups=groups, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU6(inplace=True)
    )


# ═════════════════════════════════════════════════════════════════
# 3. EFFICIENT BLOCK 1D
# ═════════════════════════════════════════════════════════════════
class EfficientBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expansion_ratio=2, use_glka=False):
        super().__init__()
        self.stride      = stride
        self.use_glka    = use_glka
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expansion_ratio

        self.expand = conv_bn_relu_1d(in_ch, hidden, kernel_size=1)

        if use_glka:
            if stride == 2:
                self.dw = conv_bn_relu_1d(hidden, hidden, kernel_size=3,
                                           stride=2, padding=1, groups=hidden)
            else:
                self.dw = nn.Identity()
            self.glka = GLKA1d(hidden)
        else:
            self.dw   = conv_bn_relu_1d(hidden, hidden, kernel_size=3,
                                         stride=stride, padding=1, groups=hidden)
            self.glka = nn.Identity()

        self.project = nn.Sequential(
            nn.Conv1d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.dw(out)
        out = self.glka(out)
        out = self.project(out)
        if self.use_residual:
            return identity + out
        return out


# ═════════════════════════════════════════════════════════════════
# 4. SIMPLE_GLKA 1D (main model)
# ═════════════════════════════════════════════════════════════════
class Simple_GLKA_1D(nn.Module):
    """
    Input : (B, 1, T)  — waveform thô, T bất kỳ (recommend ≥ 1600)
    Output: (logits, features)  — giống bản 2D để tái sử dụng training loop
    """
    def __init__(self, num_classes=35, in_channels=1):
        super().__init__()

        self.stem = conv_bn_relu_1d(in_channels, 32, kernel_size=3,
                                     stride=2, padding=1)

        self.blocks = nn.Sequential(
            EfficientBlock1d(32,  32,  stride=1, expansion_ratio=2, use_glka=False),
            EfficientBlock1d(32,  64,  stride=2, expansion_ratio=2, use_glka=True),
            EfficientBlock1d(64,  64,  stride=1, expansion_ratio=2, use_glka=True),
            EfficientBlock1d(64,  128, stride=2, expansion_ratio=2, use_glka=True),
            EfficientBlock1d(128, 128, stride=1, expansion_ratio=2, use_glka=False),
            EfficientBlock1d(128, 256, stride=2, expansion_ratio=2, use_glka=False),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x        = self.stem(x)
        x        = self.blocks(x)
        x        = self.pool(x)
        features = torch.flatten(x, 1)
        logits   = self.classifier(features)
        return logits, features

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, GLKA1d):
                m.switch_to_deploy()


# ═════════════════════════════════════════════════════════════════
# 5. KIỂM TRA
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import copy

    model = Simple_GLKA_1D(num_classes=35).eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Simple_GLKA_1D")
    print(f"Params: {total / 1e6:.3f} M")

    x = torch.randn(1, 1, 16000)   # 1s @ 16kHz
    with torch.no_grad():
        out, feat = model(x)
    print(f"Input  : {x.shape}")
    print(f"Logits : {out.shape}")
    print(f"Features: {feat.shape}")

    # Verify reparam
    m2 = copy.deepcopy(model)
    m2.reparameterize()
    m2.eval()
    with torch.no_grad():
        out2, _ = m2(x)
    diff = (out - out2).abs().max().item()
    print(f"Train vs Deploy max diff: {diff:.2e}  {'✓' if diff < 1e-4 else '✗ FAIL'}")