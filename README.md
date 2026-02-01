# PyTorch vs MindSpore 模型一致性实验总结

本实验对 AlexNet、GoogLeNet、ResNet、MobileNetV2 在 PyTorch 与 MindSpore 下的实现进行对比，**主要通过相同输入下 logits 误差论证两框架的模型一致性**：当 logits 最大/平均误差在 1e-2~1e-3 量级、预测一致率接近或达到 100% 时，可认为权重转换与前向推理等效。

---

## 1. AlexNet

| 指标 | 数值 |
|------|------|
| 数据集 | ImageNet2012val |
| logits 最大误差 | 1.40e-02 |
| logits 平均误差 | 1.24e-03 |
| 预测一致率 | **100.0%** |
| PyTorch 准确度 | 56.40% |
| MindSpore 准确度 | 56.40% |

---

## 2. GoogLeNet

| 指标 | 数值 |
|------|------|
| 数据集 | ImageNet2012val |
| logits 最大误差 | 3.48e-02 |
| logits 平均误差 | 2.51e-03 |
| 预测一致率 | **99.2%** |
| PyTorch 准确度 | 60.80% |
| MindSpore 准确度 | 60.80% |

---

## 3. MobileNetV2

| 指标 | 数值 |
|------|------|
| 数据集 | ImageNet2012val |
| logits 最大误差 | 2.57e-02 |
| logits 平均误差 | 3.17e-03 |
| 预测一致率 | **100.0%** |
| PyTorch 准确度 | 69.20% |
| MindSpore 准确度 | 69.20% |

---

## 4. ResNet

| 指标 | 数值 |
|------|------|
| 数据集 | **CIFAR-10** 测试集 |
| logits 最大误差 | 2.42e-02 |
| logits 平均误差 | 3.56e-03 |
| PyTorch 准确率 | 91.09% |
| MindSpore 准确率 | 91.09% |

---

## 结论

- **logits 误差**：四模型在两框架下的 logits 最大误差均在 1e-2~3e-2 量级、平均误差在 1e-3 量级，属浮点运算正常差异。
- **预测一致**：AlexNet、MobileNetV2 预测一致率 100%，GoogLeNet 99.2%，Top-1 预测高度一致。
- **准确度一致**：各模型 PT/MS Top-1 准确度完全相同，验证权重转换与推理逻辑正确。

综合上述结果，可认为 PyTorch 与 MindSpore 的模型实现及权重转换具有良好一致性。
