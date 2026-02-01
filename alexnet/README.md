# AlexNet Torch vs MindSpore 对比实验

## 结构说明

- MindSpore `AlexNet` 与 `torchvision.models.alexnet` 结构一致
- 预训练权重：`checkpoints/alexnet-owt-7be5be79.pth`（ImageNet 1K）
- AlexNet 无 BN，pth→ckpt 直接参数名映射

## 用法

### 1. pth 转 ckpt（首次或 pth 更新时）

```bash
python alexnet/pth_to_ckpt.py
```

### 2. 对比实验（logits 误差 + 预测一致率）

```bash
python alexnet/compare_torch_ms.py [--num-samples 100] [--skip-convert] [--use-random]
```

- `--num-samples`：样本数（默认 100）
- `--skip-convert`：跳过 pth→ckpt 转换
- `--use-random`：用随机输入（默认用 CIFAR10 测试集，自动下载到 `data/`）

## 预期结果（CIFAR10 50 样本）

- logits 最大误差：约 1e-2 量级（真实图像略大于随机）
- 预测一致率：100%（同一输入下 PT 与 MS 输出类别一致）

注：AlexNet 为 ImageNet 预训练（1000 类），此处仅用 CIFAR10 作输入来源，不计算 CIFAR10 准确度。
