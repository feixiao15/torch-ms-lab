# MobileNetV2 Torch vs MindSpore 对比实验

## 结构说明

- MindSpore `MobileNetV2` 与 `torchvision.models.mobilenet_v2` 结构一致
- 预训练权重：`mobilenet_v2-b0353104.pth`（ImageNet 1K）
- 测试输入：**ImageNet 验证集**（默认 500 张，Resize 256 + CenterCrop 224 + ImageNet 归一化）

## 用法

### 1. pth 转 ckpt

```bash
python mobilenet/pth_to_ckpt.py
```

### 2. 对比实验（需 ImageNet：`imagenet/ILSVRC2012_img_val/` + `imagenet/ILSVRC2012_validation_ground_truth.txt` + `imagenet/meta.mat`）

```bash
python mobilenet/compare_torch_ms.py [--num-samples 500] [--skip-convert] [--imagenet-dir DIR]
```

- `--num-samples`：样本数（默认 500）
- `--imagenet-dir`：图片目录（默认自动找 `ILSVRC2012_img_val`）
- `--ground-truth`：标签文件
- `--meta-mat`：meta.mat 路径（ILSVRC_ID -> torchvision index 映射）

## 预期结果（ImageNet 500 张）

- logits 最大误差：约 1e-2 量级
- logits 平均误差：约 1e-3 量级
- 预测一致率：接近 100%
- PT/MS Top-1 准确度一致
