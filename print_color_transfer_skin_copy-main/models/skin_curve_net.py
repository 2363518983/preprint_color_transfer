# models/skin_curve_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkinCurveNet(nn.Module):
    def __init__(self, n_bins=256):
        super().__init__()
        self.n_bins = n_bins
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        # 不再使用 AdaptiveAvgPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)   # 全图 GAP，作为辅助分支
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 3 * n_bins),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x, mask=None):
        x = x.clamp(0.0, 1.0)
        feat = self.features(x)               # (B, 256, H/16, W/16)

        # 全图 GAP
        global_feat = self.gap(feat).flatten(1)   # (B, 256)

        # 肤色区域 GAP（mask 下采样到 feat 分辨率）
        if mask is not None:
            m = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")  # (B,1,h,w)
            m = m.clamp(0, 1)
            # 保证分母不为零
            denom = m.sum(dim=(2,3), keepdim=True).clamp(min=1.0)
            skin_feat = (feat * m).sum(dim=(2,3), keepdim=True) / denom
            skin_feat = skin_feat.flatten(1)   # (B, 256)
        else:
            skin_feat = global_feat

        # 两个分支拼接
        feat_all = torch.cat([global_feat, skin_feat], dim=1)
        raw = self.fc(feat_all).view(-1, 3, self.n_bins)

        delta = torch.softmax(raw, dim=2)
        curve = torch.cumsum(delta, dim=2)
        cur0 = curve[:, :, :1]
        cur1 = curve[:, :, -1:]
        rng = (cur1 - cur0).clamp(min=1e-4)
        curve = (curve - cur0) / rng
        return curve.clamp(0.0, 1.0)


def apply_lut(img, curves):
    """
    可微 LUT 应用
    img: (B,3,H,W) float32 [0,1]
    curves: (B,3,n_bins)
    return: (B,3,H,W) float32 [0,1]
    """
    img = img.clamp(0.0, 1.0)
    B, C, H, W = img.shape
    n_bins = curves.shape[-1]

    # 图像值映射到 [0, n_bins-1]
    x = img * (n_bins - 1)
    x0 = x.floor().long()
    x1 = (x0 + 1).clamp(0, n_bins - 1)
    w = (x - x0.float()).clamp(0, 1)

    # 将曲线展开，按索引取值
    flat_curves = curves.reshape(B * C, n_bins)      # (B*C, n_bins)
    flat_idx0 = x0.reshape(B * C, -1)               # (B*C, H*W)
    flat_idx1 = x1.reshape(B * C, -1)

    v0 = flat_curves.gather(1, flat_idx0).reshape(B, C, H, W)
    v1 = flat_curves.gather(1, flat_idx1).reshape(B, C, H, W)

    out = v0 * (1 - w) + v1 * w
    return out