# -*- coding: utf-8 -*-
"""MindSpore AlexNet，结构匹配 torchvision.models.alexnet。"""
from mindspore import nn


class AlexNet(nn.Cell):
    """AlexNet，与 torchvision 结构一致：features + avgpool + classifier。"""

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, pad_mode="pad", padding=2, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid"),
            nn.Conv2d(64, 192, kernel_size=5, pad_mode="pad", padding=2, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid"),
            nn.Conv2d(192, 384, kernel_size=3, pad_mode="pad", padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, pad_mode="pad", padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode="pad", padding=1, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid"),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, num_classes),
        )
        self.num_classes = num_classes

    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
