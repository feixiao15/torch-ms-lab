# GoogLeNet Torch vs MindSpore 对比实验

## 结构说明

- MindSpore `GoogLeNet` 与 `torchvision.models.googlenet` 主路径一致（无 aux）
- 预训练权重：`googlenet-1378be20.pth`（ImageNet 1K）
- 测试输入：**ImageNet**（默认取 500 张，Resize 224 + CenterCrop 224，[0,1]，与 PT transform_input 一致）
- **logits 对比仅用主 logits**（aux1/aux2 为训练辅助，推理不输出）

## 用法

### 1. pth 转 ckpt

```bash
python googlenet/pth_to_ckpt.py
```

### 2. 对比实验（需 ImageNet 验证集：`imagenet/ILSVRC2012_img_val/` + `imagenet/ILSVRC2012_validation_ground_truth.txt`）

```bash
python googlenet/compare_torch_ms.py [--num-samples 500] [--skip-convert] [--imagenet-dir DIR] [--ground-truth FILE]
```

- `--num-samples`：样本数（默认 500）
- `--imagenet-dir`：图片目录（默认自动找 `ILSVRC2012_img_val` / `ILSVRC2013_DET_val` / `val`）
- `--ground-truth`：标签文件（默认 `ILSVRC2012_validation_ground_truth.txt`，第 i 行对应图像 i）

## 预期结果（ImageNet 500 张）

- 主 logits 最大误差：约 2.7e-2 量级
- 主 logits 平均误差：约 3.8e-3 量级
- 预测一致率：接近 100%
