# 修改后代码使用说明

## 本次修改汇总（对应之前提出的 7 个问题）

| 优先级 | 问题 | 修改位置 | 状态 |
|---|---|---|---|
| P0 | 拉伸变形（未中心裁剪） | `dataset.py` `center_crop_square` | ✅ |
| P0 | 验证只看 RGB MAE，无法对齐 LOSO | `train.py` `evaluate` 增加 CMYK MAE | ✅ |
| P1 | 推理端无尺寸归一化 | `inference.py` 用 `center_crop_square` + resize | ✅ |
| P1 | 训练效率低（反复读大图） | `prepare_cache.py` 离线缓存 + `dataset` 支持缓存 | ✅ |
| P2 | 缺 Lab ΔE 损失 | `losses.py` `LabLoss` + `train.py` 接入 | ✅ |
| P2 | `torch.meshgrid indexing=` 兼容性 | `color_space.py` 改手动广播 | ✅ |
| P2 | 权重初始化/BN 小 batch | `residual_cnn.py` Kaiming 初始化 | ✅ |

## 目录结构

```
print_color_transfer/
├── config.py               # 配置（含 CACHE_DIR、LAB_WEIGHT、IMG_SIZE）
├── dataset.py              # 数据集（中心裁剪 + 离线缓存）
├── train.py                # 三阶段训练（LUT / 残差CNN / 联合）
├── inference.py            # 推理（与训练同款中心裁剪）
├── prepare_cache.py        # 离线缓存生成（训练前先跑）
├── test_core.py            # 无torch环境的核心逻辑测试
├── verify_all.py           # 有torch环境的完整验证（用户环境跑）
├── models/
│   ├── lut_3d.py           # 3D LUT（恒等初始化，三线性采样）
│   └── residual_cnn.py     # 轻量残差 UNet（±0.05 限制）
└── utils/
    ├── color_space.py      # RGB<->CMYK, RGB->Lab
    └── losses.py           # LabLoss + PerceptualLoss
```

## 快速开始

### 1. 验证代码（强烈建议先跑）

```bash
# 有 torch 的环境（用户机器）：
cd print_color_transfer
python verify_all.py
# 预期：全部 9 项 OK

# 无 torch 的环境（当前沙盒）：
python test_core_standalone.py
```

### 2. 生成离线缓存（约 10-20 分钟，只需一次）

```bash
python prepare_cache.py
# 把 6000×4000 原图中心裁剪 + resize 到 512×512，存为 float16 npy
# 之后训练直接读小图，速度提升 ~10x
```

### 3. 训练

```bash
python train.py
```

训练流程：
- **阶段 1**（30 epoch）：只训 LUT，验证打印 `RGB MAE` 和 `CMYK MAE`
- **阶段 2**（30 epoch）：冻结 LUT，训残差 CNN，输入为 `(orig_cmyk, lut_cmyk)` 拼接
- **阶段 3**（10 epoch）：联合微调

**关键验收线**（对齐 LOSO）：
- LUT 阶段 `Val CMYK MAE` 应接近 **0.114**（Global 基线）
- 最终 `Val CMYK MAE` 应 **≤ 0.110**（Oracle 上限 0.1099）
- 若最终 CMYK MAE 与 LUT 阶段几乎相同 → 残差 CNN 没学到，可去掉（印证路线Ⅰ）

### 4. 推理

```bash
python inference.py test.jpg output.jpg
```

推理与训练完全对齐：同一 `center_crop_square` + 512 resize，同一 LUT/残差串联。

## 重要提醒

1. **`DATA_DIR`** 改为你真实数据路径（默认 `/home/admin/picture_data/clean_out/clean`）
2. **`NUM_WORKERS=0`**（默认），多进程 dataloader 若有 pickle 问题就保持 0
3. **验证指标以 CMYK MAE 为准**，RGB MAE 仅供参考，不要与实验A的 0.0326 比较
4. 如果显存不足，调小 `LUT_BATCH_SIZE` / `RESNET_BATCH_SIZE`
5. 当前 pipeline 未接入 ICC 色彩管理（RGB 直接当 sRGB）。若数据集混有 AdobeRGB 等，需先统一到 sRGB——可后续在 `dataset._load_or_cache` 中加入 embedded ICC 探测（复用之前 `auto_rgb_profile` 逻辑）

## 验收线对照表

| 阶段 | 指标 | 目标值 | 含义 |
|---|---|---|---|
| LUT 最佳 | CMYK MAE | ~0.114 | 全局曲线基线（对齐 LOSO Global） |
| 最终 | CMYK MAE | ≤ 0.110 | 达到 Oracle 上限（对齐 LOSO） |
| 实验A | 像素 <2% 比例 | 69.2% | 曲线可覆盖比例（参考） |

若最终 CMYK MAE 无法降到 0.110 以下，说明 30% 残差是随机噪声、非结构化，**残差 CNN 收益有限**，纯 LUT 即可交付。
