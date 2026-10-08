# 印刷调色预测（CurvePredictor + Web 审阅）

基于 **全局 RGB 曲线先验 + 参数化曲线预测器** 的印刷色差校正方案。支持命令行训练/推理，以及带批次审阅、CMYK 手工曲线与 PDF 分层保结构的 Web 服务。

交付色彩空间默认使用 **PSOcoated_v3**（`utils/PSOcoated_v3.icc`）。

## 目录结构

```
├── app.py                  # Web 服务（批次上传、推理、审阅、CMYK 曲线）
├── processor.py            # 推理核心：混合管线、ICC、PDF、CMYK 曲线
├── config.py               # 训练与推理统一配置
├── dataset.py              # 数据集（中心裁剪 + 可选离线缓存）
├── prepare_cache.py        # 生成训练用 float16 缓存
├── train.py                # 阶段 1：全局 Curve1D
├── train_curve_pred.py     # 阶段 2：CurvePredictor（融合全局曲线）
├── train_hybrid.py         # 可选：Curve1D + 残差 U-Net
├── train_rgb.py            # RGB 曲线训练（辅助脚本）
├── inference.py            # 单图 CLI 推理（Curve1D）
├── infer_hybrid.py         # 混合模型对比推理
├── eval_curve_pred.py      # 曲线预测器评估
├── export_curve.py         # 导出 ACV/NPY/CSV 曲线
├── export_curve_pred.py    # 导出预测器相关产物
├── convert_to_cmyk.py      # CMYK 转换工具
├── test_core.py            # 无 GPU 的核心逻辑测试
├── verify_all.py           # 完整环境验证（需 PyTorch）
├── models/
│   ├── curve_1d.py         # 1D RGB 曲线
│   ├── curve_predictor.py  # 图像→曲线（全局先验凸组合）
│   ├── skin_curve_net.py   # 肤色 LUT 修正（可选）
│   ├── residual_unet.py    # 残差 U-Net（可选）
│   ├── lut_3d.py           # 3D LUT（历史实验）
│   └── residual_cnn.py     # CMYK 残差 CNN（历史实验）
├── utils/                  # 色彩空间、ICC、损失函数
├── web/                    # 前端静态资源
├── scripts/                # 环境安装与启动脚本
├── checkpoints/            # 模型权重（需自行训练或放置）
└── web_data/               # Web 批次数据（运行时生成）
```

## 环境

```bash
# 新建虚拟环境并安装依赖
./scripts/setup_env.sh

# 或复用已有 conda 环境
./scripts/setup_env.sh --conda 你的环境名

pip install -r requirements.txt
```

主要依赖：`torch`、`numpy`、`pillow`、`scikit-image`、`pypdfium2`、`pypdf`。

## 数据与配置

1. 在 `config.py` 中设置 `DATA_DIR` 为成对样本目录（命名约定见 `dataset.py` / `train_hybrid.py` 的 `*_input` / `*_target` 后缀）。
2. 大图训练建议先跑缓存：

```bash
python prepare_cache.py
```

3. 全局曲线先验需存在于 `checkpoints/global_curve.pt`（由阶段 1 训练或导出产生），CurvePredictor 加载时会强制校验。

## 训练流程（推荐顺序）

| 阶段 | 脚本 | 产物 |
|------|------|------|
| 1 | `python train.py` | 全局 RGB 曲线 → `checkpoints/` |
| 2 | `python train_curve_pred.py` | `curve_pred_best.pth` |
| 可选 | `python train_hybrid.py` | `unet_best.pth` 等 |
| 可选 | 肤色模型训练脚本 | `skin_curve_best.pth`（与主 checkpoint 同目录） |

`processor.load_model()` 会在主权重同目录下自动尝试加载 `skin_curve_best.pth`、`unet_best.pth`，组成 **HybridPipeline**（全局曲线 → 肤色曲线 → U-Net 残差）。

验证：

```bash
python test_core.py      # 轻量检查
python verify_all.py     # 需完整 PyTorch 环境
python eval_curve_pred.py
```

## Web 服务

```bash
./scripts/start_app.sh
# 或
python app.py --host 0.0.0.0 --port 5001 --model checkpoints/curve_pred_best.pth --data web_data
```

功能概要：

- 批次创建、多图/PDF 上传、后台推理队列
- 输出 **CMYK 印刷稿**（TIFF/PDF）与 **sRGB 软打样预览**
- PDF 优先 **保留分层** 替换嵌入图；失败时回退整页栅格重组
- 批次级 **CMYK 手工曲线**（主曲线 + C/M/Y/K），实时预览与整批重渲染
- 可选上传目标图，计算 ΔE00 / MAE
- 按审阅状态打包 ZIP 下载

Mac 部署可参考 `scripts/deploy_mac.sh`、`scripts/install_autostart.sh`。

## 命令行推理

```bash
# Curve1D / 旧版单图推理
python inference.py --image input.jpg --save_dir out/

# 混合模型三路对比
python infer_hybrid.py --image input.jpg --use_unet
```

生产 Web 与批量处理统一走 `processor.process_file()`，与 CLI 共用同一套 ICC 与混合推理逻辑。

## 指标说明

- **训练/验证**：曲线阶段以 RGB 与 Lab 相关损失为主；CMYK 评估需经 ICC 往返，与 LOSO 实验对齐时以 **CMYK MAE / ΔE** 为准。
- **Web 审阅**：有目标图时在 sRGB 空间计算 **ΔE00** 与 **MAE**。

## 常见问题

1. **显存不足**：减小 `config.py` 中 `PRED_BATCH_SIZE`、`CURVE_BATCH_SIZE`；Web 端大图会自动分块推理（见 `processor.py` 中 `TILE_*` 常量）。
2. **PDF 过大**：已提高 `pypdf` 流大小上限；分层失败时会打印日志并回退 TIFF 页重组。
3. **输入 ICC**：嵌入 PDF 图或带 ICC 的 TIFF 会优先用 embedded profile 转 sRGB，再进模型；无 ICC 的 CMYK 走 PSOcoated_v3。
