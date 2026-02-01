# -*- coding: utf-8 -*-
"""
ResNet50 图像分类 - MindSpore 复现
基于 CIFAR-10 数据集，在 CPU 上训练
参考: https://www.mindspore.cn/tutorials/zh-CN/r2.8.0/cv/resnet50.html
"""

import os
import urllib.request
import tarfile
from typing import Type, Union, List, Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal

# ============ 设备配置：使用 CPU 训练 ============
ms.set_context(device_target="CPU")

# ============ 数据集下载 ============
def download_cifar10(data_dir: str = "./datasets-cifar10-bin"):
    """下载并解压 CIFAR-10 二进制数据集"""
    url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-binary.tar.gz")
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, "cifar-10-batches-bin")):
        print("正在下载 CIFAR-10 数据集...")
        urllib.request.urlretrieve(url, tar_path)
        print("正在解压...")
        with tarfile.open(tar_path, "r:gz") as f:
            f.extractall(data_dir)
        if os.path.exists(tar_path):
            os.remove(tar_path)
        print("数据集准备完成。")
    return os.path.join(data_dir, "cifar-10-batches-bin")


# ============ 数据集加载与预处理 ============
def create_dataset_cifar10(dataset_dir: str, usage: str, resize: int, batch_size: int, workers: int):
    """创建 CIFAR-10 数据管道"""
    data_set = ds.Cifar10Dataset(
        dataset_dir=dataset_dir,
        usage=usage,
        num_parallel_workers=workers,
        shuffle=True,
    )
    trans = []
    if usage == "train":
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5),
        ]
    trans += [
        vision.Resize((resize, resize)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW(),
    ]
    target_trans = transforms.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=workers)
    data_set = data_set.map(operations=target_trans, input_columns="label", num_parallel_workers=workers)
    data_set = data_set.batch(batch_size)
    return data_set


# ============ 残差块与 ResNet50 网络 ============
weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)


class ResidualBlockBase(nn.Cell):
    """Building Block，适用于 ResNet18/34"""

    expansion: int = 1

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super(ResidualBlockBase, self).__init__()
        self.norm = norm if norm else nn.BatchNorm2d(out_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, weight_init=weight_init)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, weight_init=weight_init)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm(out)
        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


class ResidualBlock(nn.Cell):
    """Bottleneck，适用于 ResNet50/101/152"""

    expansion = 4

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, weight_init=weight_init)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, weight_init=weight_init)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(
            out_channel, out_channel * self.expansion, kernel_size=1, weight_init=weight_init
        )
        self.norm3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


def make_layer(
    last_out_channel: int,
    block: Type[Union[ResidualBlockBase, ResidualBlock]],
    channel: int,
    block_nums: int,
    stride: int = 1,
):
    """堆叠残差块"""
    down_sample = None
    if stride != 1 or last_out_channel != channel * block.expansion:
        down_sample = nn.SequentialCell([
            nn.Conv2d(
                last_out_channel,
                channel * block.expansion,
                kernel_size=1,
                stride=stride,
                weight_init=weight_init,
            ),
            nn.BatchNorm2d(channel * block.expansion, gamma_init=gamma_init),
        ])
    layers = [block(last_out_channel, channel, stride=stride, down_sample=down_sample)]
    in_channel = channel * block.expansion
    for _ in range(1, block_nums):
        layers.append(block(in_channel, channel))
    return nn.SequentialCell(layers)


class ResNet(nn.Cell):
    """ResNet 主干"""

    def __init__(
        self,
        block: Type[Union[ResidualBlockBase, ResidualBlock]],
        layer_nums: List[int],
        num_classes: int,
        input_channel: int,
    ) -> None:
        super(ResNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, weight_init=weight_init)
        self.norm = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = make_layer(64, block, 64, layer_nums[0])
        self.layer2 = make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=2)
        self.layer3 = make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
        self.layer4 = make_layer(256 * block.expansion, block, 512, layer_nums[3], stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(in_channels=input_channel, out_channels=num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def resnet50(num_classes: int = 10):
    """ResNet50 模型（CIFAR-10 默认 10 类）"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes, input_channel=2048)


# ============ 训练与评估 ============
def train_step(network, opt, loss_fn, inputs, targets):
    """单步训练"""
    def forward_fn(inputs, targets):
        logits = network(inputs)
        return loss_fn(logits, targets)

    grad_fn = ms.value_and_grad(forward_fn, None, opt.parameters)
    loss, grads = grad_fn(inputs, targets)
    opt(grads)
    return loss


def train_epoch(network, data_loader, opt, loss_fn, epoch, num_epochs, step_size_train):
    """一个 epoch 训练"""
    losses = []
    network.set_train(True)
    for i, (images, labels) in enumerate(data_loader):
        loss = train_step(network, opt, loss_fn, images, labels)
        losses.append(float(loss.asnumpy()))
        if i % 50 == 0 or i == step_size_train - 1:
            print(
                "Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.3f]"
                % (epoch + 1, num_epochs, i + 1, step_size_train, loss)
            )
    return sum(losses) / len(losses)


def evaluate(network, data_loader):
    """验证集评估"""
    network.set_train(False)
    correct_num = 0.0
    total_num = 0.0
    for images, labels in data_loader:
        logits = network(images)
        pred = logits.argmax(axis=1)
        correct = ops.equal(pred, labels).reshape((-1,))
        correct_num += correct.sum().asnumpy()
        total_num += correct.shape[0]
    return float(correct_num / total_num)


# ============ 主程序 ============
def main():
    # 超参数
    data_dir = "./datasets-cifar10-bin"
    batch_size = 32  # CPU 上可适当减小
    image_size = 32
    workers = 4
    num_classes = 10
    num_epochs = 5  # 可改为 80 以获得更好效果

    # 下载并获取数据路径
    dataset_dir = download_cifar10(data_dir)

    # 创建数据集
    dataset_train = create_dataset_cifar10(
        dataset_dir=dataset_dir,
        usage="train",
        resize=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    step_size_train = dataset_train.get_dataset_size()

    dataset_val = create_dataset_cifar10(
        dataset_dir=dataset_dir,
        usage="test",
        resize=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    step_size_val = dataset_val.get_dataset_size()

    # 构建网络
    network = resnet50(num_classes=num_classes)

    # 学习率与优化器
    lr = nn.cosine_decay_lr(
        min_lr=1e-5,
        max_lr=1e-3,
        total_step=step_size_train * num_epochs,
        step_per_epoch=step_size_train,
        decay_epoch=num_epochs,
    )
    opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # 保存目录
    best_ckpt_dir = "./BestCheckpoint"
    best_ckpt_path = os.path.join(best_ckpt_dir, "resnet50-best.ckpt")
    os.makedirs(best_ckpt_dir, exist_ok=True)

    best_acc = 0.0
    data_loader_train = dataset_train.create_tuple_iterator(num_epochs=num_epochs)
    data_loader_val = dataset_val.create_tuple_iterator(num_epochs=num_epochs)

    print("使用设备: CPU")
    print("Start Training Loop ...")

    for epoch in range(num_epochs):
        curr_loss = train_epoch(
            network, data_loader_train, opt, loss_fn, epoch, num_epochs, step_size_train
        )
        curr_acc = evaluate(network, data_loader_val)

        print("-" * 50)
        print(
            "Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]"
            % (epoch + 1, num_epochs, curr_loss, curr_acc)
        )
        print("-" * 50)

        if curr_acc > best_acc:
            best_acc = curr_acc
            ms.save_checkpoint(network, best_ckpt_path)
            print("已保存最佳模型: %s" % best_ckpt_path)

    print("=" * 60)
    print("训练结束。最佳验证准确率: %.3f" % best_acc)
    print("最佳模型已保存至: %s" % best_ckpt_path)


if __name__ == "__main__":
    main()
