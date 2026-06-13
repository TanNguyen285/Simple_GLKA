import torch
import torch.nn as nn
import torch.nn.functional as F


class GLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.K = 13

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, dilation=1),
            nn.BatchNorm2d(dim)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=3, groups=dim, dilation=3),
            nn.BatchNorm2d(dim)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=4, groups=dim, dilation=2),
            nn.BatchNorm2d(dim)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=6, groups=dim, dilation=3),
            nn.BatchNorm2d(dim)
        )

        self.reparam_conv = None

    def forward(self, x):
        # Conv5x5 depthwise tạo feature map chung cho cả 4 branch
        global_conv = self.conv0(x)
        # SE attention gate
        anchor = global_conv * self.se(global_conv)
        
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
            padding=self.K // 2,
            groups=self.dim,
            bias=True
        )
        self.reparam_conv.weight.data = W_equiv
        self.reparam_conv.bias.data   = B_equiv
        
        del self.branch1, self.branch2, self.branch3, self.branch4

    def _fuse_bn(self, sequential_block):
        conv = sequential_block[0]
        bn   = sequential_block[1]
        std  = (bn.running_var + bn.eps).sqrt()
        t    = (bn.weight / std).reshape(-1, 1, 1, 1)
        # [FIX 1] công thức đúng: không bỏ conv.bias
        b_conv  = conv.bias if conv.bias is not None else \
                  torch.zeros(conv.out_channels, device=conv.weight.device)
        w_fused = conv.weight * t
        b_fused = bn.bias + (b_conv - bn.running_mean) * bn.weight / std
        return w_fused, b_fused

    def _to_target_k(self, kernel, d):
        c, m, orig_k, _ = kernel.shape
        kd = (orig_k - 1) * d + 1
        # [FIX 2] tạo thẳng tensor K×K, căn giữa đúng — không dùng F.pad
        out    = torch.zeros((c, m, self.K, self.K),
                             device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset : offset + kd : d,
                   offset : offset + kd : d] = kernel
        return out


# =================================================================
# Các thành phần bổ trợ
# =================================================================
def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class EfficientBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio=2, use_glka=False):
        super(EfficientBlock, self).__init__()
        self.stride       = stride
        self.use_glka     = use_glka
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim        = in_channels * expansion_ratio

        self.expand = conv_bn_relu(in_channels, hidden_dim, kernel_size=1)
        
        if self.use_glka:
            # stride=2 → depthwise riêng để downsample, sau đó GLKA
            # stride=1 → GLKA trực tiếp
            self.dw   = conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3,
                                     stride=2, padding=1,
                                     groups=hidden_dim) if stride == 2 \
                        else nn.Identity()
            self.glka = GLKA(hidden_dim)
        else:
            self.dw   = conv_bn_relu(hidden_dim, hidden_dim, kernel_size=3,
                                     stride=stride, padding=1, groups=hidden_dim)
            self.glka = nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.dw(out)
        out = self.glka(out)
        out = self.project(out)
        return identity + out if self.use_residual else out


# =================================================================
# Kiến trúc tổng thể Simple_GLKA
# =================================================================
class Simple_GLKA(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple_GLKA, self).__init__()
        
        self.stem = conv_bn_relu(3, 32, kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            EfficientBlock(32,  32,  stride=1, expansion_ratio=2, use_glka=False),
            EfficientBlock(32,  64,  stride=2, expansion_ratio=2, use_glka=True),
            EfficientBlock(64,  64,  stride=1, expansion_ratio=2, use_glka=True),
            EfficientBlock(64,  128, stride=2, expansion_ratio=2, use_glka=True),
            EfficientBlock(128, 128, stride=1, expansion_ratio=2, use_glka=False),
            EfficientBlock(128, 256, stride=2, expansion_ratio=2, use_glka=False),
        )

        self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_fc   = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x        = self.stem(x)
        x        = self.blocks(x)
        x        = self.classifier_pool(x)
        features = torch.flatten(x, 1)
        out      = self.classifier_fc(features)
        return out, features


# =================================================================
# Kiểm tra model + verify train vs deploy
# =================================================================
if __name__ == "__main__":
    import copy

    model = Simple_GLKA(num_classes=3)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Kiến trúc : Simple_GLKA")
    print(f"Tham số   : {total_params / 1e6:.3f} M")

    test_input = torch.randn(1, 3, 224, 224)

    # [FIX 3] populate BN running stats trước khi verify deploy
    # (BN stats = 0/1 mặc định → switch_to_deploy cho output sai dù công thức đúng)
    print("\nPopulating BN stats...")
    model.train()
    with torch.no_grad():
        for _ in range(20):
            model(torch.randn(4, 3, 224, 224))
    model.eval()

    with torch.no_grad():
        out, features = model(test_input)

    print(f"Input     : {test_input.shape}")
    print(f"Logits    : {out.shape}")
    print(f"Features  : {features.shape}")

    # Verify train vs deploy
    model_deploy = copy.deepcopy(model)
    for m in model_deploy.modules():
        if isinstance(m, GLKA):
            m.switch_to_deploy()

    with torch.no_grad():
        out_deploy, _ = model_deploy(test_input)

    diff = (out - out_deploy).abs().max().item()
    print(f"\nTrain vs Deploy max diff: {diff:.2e}  "
          f"{'✅ OK' if diff < 1e-4 else '❌ FAIL'}")

    # Đếm GLKA đã được deploy
    deployed = sum(
        1 for m in model_deploy.modules()
        if isinstance(m, GLKA) and m.reparam_conv is not None
    )
    print(f"GLKA blocks deployed    : {deployed}")