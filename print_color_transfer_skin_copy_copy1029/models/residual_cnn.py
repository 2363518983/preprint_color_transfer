# -*- coding: utf-8 -*-
"""轻量残差 CNN：在曲线输出上做小幅空间修正。

输入 = cat(input_RGB(3), curve_out_CMYK(4)) = 7 通道
输出 = (B,4,H,W)，量级约 0.05
"""
import torch
import torch.nn as nn


class ResidualCNN(nn.Module):
    def __init__(self, in_channels=7, out_channels=4, base_filters=32, n_blocks=3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_filters, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n_blocks - 1):
            layers += [nn.Conv2d(base_filters, base_filters, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(base_filters, out_channels, 1)]
        self.net = nn.Sequential(*layers)
        self.scale = 0.05

    def forward(self, inp, curve_out):
        x = torch.cat([inp, curve_out], dim=1)   # (B,7,H,W)
        return self.scale * self.net(x)