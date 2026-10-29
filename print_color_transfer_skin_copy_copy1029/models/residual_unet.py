# -*- coding: utf-8 -*-
"""
残差 U-Net v3（空间门控版）

改动（相比 v2）：
- ★ 移除全局标量 residual_gate
- ★ 新增空间门控头 gate_head：per-pixel 门控 (B,1,H,W)
    初始 bias = -4.6 → sigmoid(-4.6) ≈ 0.01 → 初始残差 std≈0.0067
- ★ forward 支持 return_gate=True，训练时可拿到 gate 做正则
- 向后兼容：默认 forward(x) 只返回 output tensor，与旧接口一致
  （processor.py 不需要改）

★ 注意：结构已变，加载旧的 unet_best.pth 会报错，重训前请备份。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            _, _, h, w = x.shape
            skip = F.interpolate(skip, size=(h, w), mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResidualUNet(nn.Module):
    """输入 (B,6,H,W) → 输出 (B,3,H,W)。
    ★ 输出残差被 per-pixel 空间门控缩放，初始门控≈0.01，安全接入主链路。
    """

    def __init__(self, in_ch=6, base=48, out_ch=3):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4

        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = Down(c1, c2)
        self.enc3 = Down(c2, c3)
        self.bottleneck = Down(c3, c3)

        self.up3 = Up(c3, c3, c2)
        self.up2 = Up(c2, c2, c1)
        self.up1 = Up(c1, c1, c1)

        # 残差头
        self.out_conv = nn.Conv2d(c1, out_ch, 1)
        nn.init.kaiming_normal_(self.out_conv.weight, mode="fan_in", nonlinearity="linear")

        # ★ 空间门控头：per-pixel，初始 ≈ 0.01
        self.gate_head = nn.Conv2d(c1, 1, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -4.6)   # sigmoid(-4.6) ≈ 0.00996

    def forward(self, x, return_gate=False):
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        b = self.bottleneck(c3)

        u3 = self.up3(b, c3)
        u2 = self.up2(u3, c2)
        u1 = self.up1(u2, c1)

        raw = self.out_conv(u1)                     # (B,3,H,W)，std≈0.67
        gate = torch.sigmoid(self.gate_head(u1))    # (B,1,H,W) ∈ (0,1)
        out = raw * gate

        if return_gate:
            return out, gate
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    unet = ResidualUNet(in_ch=6, base=48, out_ch=3)
    x = torch.rand(2, 6, 256, 256)
    out, gate = unet(x, return_gate=True)
    print(f"输出 shape={tuple(out.shape)}, std={out.std().item():.6f}")
    print(f"gate mean={gate.mean().item():.4f}, min={gate.min().item():.4f}, max={gate.max().item():.4f}")

    loss = out.abs().mean()
    loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in unet.parameters())
    print(f"梯度流通: {'✅' if grad_ok else '❌'}")
    # 期望 std ≈ 0.67 * 0.01 ≈ 0.0067