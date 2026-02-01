# -*- coding: utf-8 -*-
"""MindSpore GoogLeNet，结构匹配 torchvision.models.googlenet，仅主 logits。"""
from mindspore import nn, ops
from mindspore import Tensor


class BasicConv2d(nn.Cell):
    """Conv2d(no bias) + BatchNorm2d(eps=0.001) + ReLU，匹配 PT BasicConv2d。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, pad_mode="pad", padding=padding, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Inception(nn.Cell):
    """Inception 模块，4 分支。"""

    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int, ch5x5red: int, ch5x5: int, pool_proj: int) -> None:
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.SequentialCell(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.SequentialCell(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )
        self.branch4 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=3, stride=1, pad_mode="same"),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def construct(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return ops.concat([b1, b2, b3, b4], axis=1)


class GoogLeNet(nn.Cell):
    """GoogLeNet，仅主 logits（推理用，无 aux）。与 PT 预训练一致，输入 [0,1] 且做 transform_input。"""

    def __init__(self, num_classes: int = 1000, dropout: float = 0.2, transform_input: bool = True) -> None:
        super().__init__()
        self.transform_input = transform_input
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Dense(1024, num_classes)

    def _transform_input(self, x):
        """与 PT 一致：输入 [0,1]，转为 ImageNet norm 近似。"""
        x0 = x[:, 0:1] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x1 = x[:, 1:2] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x2 = x[:, 2:3] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return ops.concat([x0, x1, x2], axis=1)

    def construct(self, x):
        if self.transform_input:
            x = self._transform_input(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
