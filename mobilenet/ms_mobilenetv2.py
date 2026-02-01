from mindspore import nn, ops
from typing import List, Optional


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(nn.SequentialCell):
    """Conv + BN + ReLU6，参数名：0=Conv, 1=BN, 2=ReLU6。"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride=stride, pad_mode="pad",
                padding=padding, group=groups, has_bias=False
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(),
        )


class InvertedResidual(nn.Cell):
    """
    Inverted Residual，参数名与 PT 一致。
    expand_ratio=1: conv=[Conv2dNormActivation(dw), Conv2d(pw), BN]
    expand_ratio≠1: conv=[Conv2dNormActivation(pw-exp), Conv2dNormActivation(dw), Conv2d(pw), BN]
    """

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Cell] = []
        if expand_ratio != 1:
            # conv.0: pw expansion (Conv2dNormActivation)
            layers.append(Conv2dNormActivation(inp, hidden_dim, kernel_size=1))
            # conv.1: dw (Conv2dNormActivation)
            layers.append(Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
            # conv.2: pw-linear (Conv2d)
            layers.append(nn.Conv2d(hidden_dim, oup, 1, stride=1, pad_mode="pad", padding=0, has_bias=False))
            # conv.3: BN
            layers.append(nn.BatchNorm2d(oup))
        else:
            # conv.0: dw (Conv2dNormActivation)
            layers.append(Conv2dNormActivation(inp, inp, stride=stride, groups=inp))
            # conv.1: pw-linear (Conv2d)
            layers.append(nn.Conv2d(inp, oup, 1, stride=1, pad_mode="pad", padding=0, has_bias=False))
            # conv.2: BN
            layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Cell):
    """MobileNetV2，参数名匹配 torchvision。"""

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = 32
        last_channel = 1280
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)

        features: List[nn.Cell] = [Conv2dNormActivation(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(Conv2dNormActivation(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.SequentialCell(features)

        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(self.last_channel, num_classes),
        )

    def construct(self, x):
        x = self.features(x)
        x = ops.mean(x, axis=(2, 3))
        x = self.classifier(x)
        return x
