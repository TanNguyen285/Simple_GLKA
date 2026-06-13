import copy
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _check_k(K: int) -> None:
    if K < 13:
        raise ValueError(f"deploy_k={K} quá nhỏ. Branch 5×5/d3 có eff=13, cần K ≥ 13.")
    if K % 2 == 0:
        raise ValueError(f"deploy_k phải là số lẻ, nhận được {K}.")


def Conv2d_block(in_ch, out_ch, k, stride=1, padding=0, groups=1):

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────
# GLKA_Block — core attention block
# ─────────────────────────────────────────────────────────────────

class GLKA_Block(nn.Module):
    _BRANCHES = [(3, 1), (3, 3), (5, 2), (5, 3)]

    def __init__(self, ch: int, deploy_k: int = 13):
        super().__init__()
        _check_k(deploy_k)
        self.ch = ch
        self.deploy_k = deploy_k

        # Anchor
        self.anchor_conv = nn.Conv2d(ch, ch, 5, padding=2, groups=ch)
        self.SE_Block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(1, ch // 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, ch // 8), ch, 1),
            nn.Sigmoid(),
        )
        # 4 dilated branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, k, padding=(k - 1) // 2 * d,
                          groups=ch, dilation=d),
                nn.BatchNorm2d(ch),
            )
            for k, d in self._BRANCHES
        ])

        self._fused = None  # None = training mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Global = self.anchor_conv(x)
        anchor = Global * self.SE_Block(Global)

        if self._fused is not None:
            attn = self._fused(Global)
        else:
            attn = sum(b(Global) for b in self.branches)

        return anchor * attn

    def reparameterize(self):
        if self._fused is not None:
            return

        K = self.deploy_k
        W = torch.zeros(self.ch, 1, K, K,
                        device=self.anchor_conv.weight.device,
                        dtype=self.anchor_conv.weight.dtype)
        b = torch.zeros(self.ch, device=W.device, dtype=W.dtype)

        for branch, (k, d) in zip(self.branches, self._BRANCHES):
            bw, bb = self._fuse_bn(branch)
            W += self._embed(bw, d, K)
            b += bb

        self._fused = nn.Conv2d(self.ch, self.ch, K,
                                padding=K // 2, groups=self.ch, bias=True)
        self._fused.weight.data = W
        self._fused.bias.data = b
        del self.branches

    @staticmethod
    def _fuse_bn(seq: nn.Sequential):
        conv, bn = seq[0], seq[1]
        std = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        b_in = conv.bias if conv.bias is not None else \
               torch.zeros(conv.out_channels, device=conv.weight.device)
        return conv.weight * scale, \
               bn.bias + (b_in - bn.running_mean) * bn.weight / std

    @staticmethod
    def _embed(kernel: torch.Tensor, d: int, K: int) -> torch.Tensor:
        c, _, k, _ = kernel.shape
        eff = (k - 1) * d + 1
        canvas = torch.zeros(c, 1, K, K, device=kernel.device, dtype=kernel.dtype)
        off = (K - eff) // 2
        canvas[:, :, off:off + eff:d, off:off + eff:d] = kernel
        return canvas


# ─────────────────────────────────────────────────────────────────
# IRB — Inverted Residual Block
# ─────────────────────────────────────────────────────────────────

class MB_Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expansion=2,
                 use_glka=False, deploy_k=13):
        super().__init__()
        self.residual = (stride == 1 and in_ch == out_ch)
        hid = in_ch * expansion

        self.expand = Conv2d_block(in_ch, hid, 1)

        if use_glka:
            self.dw = Conv2d_block(hid, hid, 3, stride=2, padding=1,
                            groups=hid) if stride == 2 else nn.Identity()
            self.attn = GLKA_Block(hid, deploy_k=deploy_k)
        else:
            self.dw = Conv2d_block(hid, hid, 3, stride=stride, padding=1, groups=hid)
            self.attn = nn.Identity()

        self.proj = nn.Sequential(
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.proj(self.attn(self.dw(self.expand(x))))
        return x + out if self.residual else out


# ─────────────────────────────────────────────────────────────────
# Sub-modules của Velocity
# ─────────────────────────────────────────────────────────────────

class GLKANet(nn.Module):
    def __init__(self, deploy_k=13):
        super().__init__()
        self.stem = Conv2d_block(3, 32, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            MB_Block(32,  32,  stride=1, use_glka=False, deploy_k=deploy_k),
            MB_Block(32,  64,  stride=2, use_glka=True,  deploy_k=deploy_k),
            MB_Block(64,  64,  stride=1, use_glka=True,  deploy_k=deploy_k),
            MB_Block(64,  128, stride=2, use_glka=True,  deploy_k=deploy_k),
        )
    def forward(self, x):
        return self.blocks(self.stem(x))
    
class MotionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = Conv2d_block(2, 16, 3, stride=2, padding=1)
        self.b1   = self._dw(16, 32,  stride=2)
        self.b2   = self._dw(32, 64,  stride=2)
        self.b3   = self._dw(64, 128, stride=1)

    @staticmethod
    def _dw(in_ch, out_ch, stride):
        return nn.Sequential(
            Conv2d_block(in_ch, out_ch, 1),
            Conv2d_block(out_ch, out_ch, 3, stride=stride, padding=1, groups=out_ch),
        )

    def forward(self, x):
        return self.b3(self.b2(self.b1(self.stem(x))))


class FusionBlock(nn.Module):
    """Concat(app, motion) → (B, 128, H/8, W/8)"""

    def __init__(self):
        super().__init__()
        self.fuse = Conv2d_block(256, 128, 1)

    def forward(self, a, m):
        return self.fuse(torch.cat([a, m], dim=1))


class LightGRU(nn.Module):
    """
    GRU tối giản: 1 update gate + 1 candidate (không có reset gate).
    Nhẹ hơn GRU chuẩn ~2×, đủ để tích lũy ngữ cảnh thời gian.

    feat_dim : chiều feat từ backbone (128)
    state_dim: chiều state tích lũy (32) — nhỏ, cố định bộ nhớ
    """

    def __init__(self, feat_dim: int = 128, state_dim: int = 32):
        super().__init__()
        in_dim = feat_dim + state_dim
        self.gate = nn.Sequential(nn.Linear(in_dim, state_dim), nn.Sigmoid())
        self.cand = nn.Sequential(nn.Linear(in_dim, state_dim), nn.Tanh())

    def forward(self, feat: torch.Tensor, state: torch.Tensor):
        """
        feat  : (B, feat_dim)
        state : (B, state_dim)
        →      new_state (B, state_dim)
        """
        x = torch.cat([feat, state], dim=1)
        z = self.gate(x)                      # update gate
        h = self.cand(x)                      # candidate
        return z * state + (1 - z) * h        # blend: giữ cũ hay nhận mới


# ─────────────────────────────────────────────────────────────────
# Velocity — dual-stream + temporal state
# ─────────────────────────────────────────────────────────────────

class Velocity(nn.Module):
    def __init__(self, num_classes: int = 4, deploy_k: int = 13, state_dim: int = 32):
        super().__init__()
        self.app    = GLKANet(deploy_k=deploy_k)
        self.motion = MotionNet()
        self.fusion = FusionBlock()
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.gru    = LightGRU(feat_dim=128, state_dim=state_dim)
        self.head   = nn.Sequential(
            nn.Linear(128 + state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self.state_dim = state_dim

    def init_state(self, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        """Gọi 1 lần khi bắt đầu video stream."""
        return torch.zeros(batch_size, self.state_dim, device=device)

    def forward(self, frame: torch.Tensor,
                motion: torch.Tensor,
                state: torch.Tensor):
        app_f  = self.app(frame)                          # (B,128,H/8,W/8)
        mot_f  = self.motion(motion)                      # (B,128,H/8,W/8)
        fused  = self.fusion(app_f, mot_f)                # (B,128,H/8,W/8)
        feat   = torch.flatten(self.pool(fused), 1)       # (B,128)
        new_st = self.gru(feat, state)                    # (B,32)
        logits = self.head(torch.cat([feat, new_st], 1))  # (B,C)
        return logits, new_st

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, GLKA_Block):
                m.reparameterize()


# ─────────────────────────────────────────────────────────────────
# SimpleNet — single-input cho ảnh tĩnh
# ─────────────────────────────────────────────────────────────────

class SimpleNet(nn.Module):
    """
    Single-input model: 1 frame → logits + embedding.
    Nhẹ hơn Velocity, dùng cho ảnh tĩnh hoặc khi không có diff.
    """

    def __init__(self, num_classes: int = 2, deploy_k: int = 13):
        super().__init__()
        K = deploy_k
        self.stem = Conv2d_block(3, 32, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            MB_Block(32,  32,  stride=1, use_glka=False, deploy_k=K),
            MB_Block(32,  64,  stride=2, use_glka=True,  deploy_k=K),
            MB_Block(64,  64,  stride=1, use_glka=True,  deploy_k=K),
            MB_Block(64,  128, stride=2, use_glka=True,  deploy_k=K),
            MB_Block(128, 128, stride=1, use_glka=False, deploy_k=K),
            MB_Block(128, 256, stride=2, use_glka=False, deploy_k=K),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        feat   = torch.flatten(self.pool(self.blocks(self.stem(x))), 1)
        logits = self.fc(self.dropout(feat))
        return logits, feat

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, GLKA_Block):
                m.reparameterize()


# ─────────────────────────────────────────────────────────────────
# Utility: tạo motion input từ 3 frame liên tiếp
# ─────────────────────────────────────────────────────────────────

def make_motion(t: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    t, t1, t2: (B, 3, H, W) — frame hiện tại, trước 1, trước 2
    return   : (B, 2, H, W) — magnitude diff stack
    """
    d1 = (t  - t1).abs().mean(1, keepdim=True)   # (B,1,H,W)
    d2 = (t1 - t2).abs().mean(1, keepdim=True)   # (B,1,H,W)
    return torch.cat([d1, d2], dim=1)             # (B,2,H,W)


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, W = 1, 224, 224
    ft  = torch.randn(B, 3, H, W)
    ft1 = torch.randn(B, 3, H, W)
    ft2 = torch.randn(B, 3, H, W)
    mot = make_motion(ft, ft1, ft2)   # (B,2,H,W)

    for K in [13, 15, 19]:
        print(f"\n── deploy_k = {K} ──")

        # ── Velocity ──
        v = Velocity(num_classes=4, deploy_k=K).eval()
        st = v.init_state(B)
        params = sum(p.numel() for p in v.parameters())
        with torch.no_grad():
            out, st2 = v(ft, mot, st)
        vd = copy.deepcopy(v)
        vd.reparameterize()
        with torch.no_grad():
            out_d, _ = vd(ft, mot, st)
        err = (out - out_d).abs().max().item()
        print(f"  Velocity  {params/1e6:.3f}M  logits{tuple(out.shape)}  "
              f"state{tuple(st2.shape)}  reparam_err {err:.2e}  "
              f"{'✓' if err < 1e-4 else '✗'}")

        # ── SimpleNet ──
        s = SimpleNet(num_classes=2, deploy_k=K).eval()
        params2 = sum(p.numel() for p in s.parameters())
        with torch.no_grad():
            out2, emb = s(ft)
        sd = copy.deepcopy(s)
        sd.reparameterize()
        with torch.no_grad():
            out2_d, _ = sd(ft)
        err2 = (out2 - out2_d).abs().max().item()
        print(f"  SimpleNet {params2/1e6:.3f}M  logits{tuple(out2.shape)}  "
              f"embed{tuple(emb.shape)}   reparam_err {err2:.2e}  "
              f"{'✓' if err2 < 1e-4 else '✗'}")

    # Guard test
    try:
        _ = Velocity(deploy_k=7)
        print("\n  K=7 guard: ✗")
    except ValueError as e:
        print(f"\n  K=7 guard: ✓  ({e})")

    # Inference loop demo
    print("\n── Inference loop demo (Velocity) ──")
    model = Velocity(num_classes=4).eval()
    state = model.init_state(batch_size=1)
    LABELS = ["thông thoáng", "đông chạy", "xả kẹt", "kẹt xe"]

    # Giả lập 5 frame liên tiếp
    frames = [torch.randn(1, 3, 224, 224) for _ in range(7)]
    for i in range(2, 7):
        motion_in = make_motion(frames[i], frames[i-1], frames[i-2])
        with torch.no_grad():
            logits, state = model(frames[i], motion_in, state)
        pred = logits.argmax(dim=1).item()
        print(f"  frame {i}  →  {LABELS[pred]} (logits: {logits[0].tolist()})")