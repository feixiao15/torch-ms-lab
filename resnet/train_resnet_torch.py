# -*- coding: utf-8 -*-
"""
PyTorch 训练 ResNet50 - CIFAR-10
用于 GPU 训练，可与 MindSpore 推理配合做性能验证。
在 cmd 中运行: python train_resnet_torch.py
"""

import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50

# ============ 配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./data"  # CIFAR-10 会下载到此目录
BATCH_SIZE = 128
NUM_EPOCHS = 5
NUM_WORKERS = 4
LR = 1e-3
NUM_CLASSES = 10
BEST_CKPT_DIR = "./BestCheckpoint_torch"
BEST_CKPT_PATH = os.path.join(BEST_CKPT_DIR, "resnet50-best.pth")

# CIFAR-10 均值和标准差（与 MindSpore 版本一致，便于后续推理对比）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders():
    """构建 CIFAR-10 训练/测试 DataLoader"""
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=test_tf
    )
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, test_loader


def build_resnet50(num_classes: int = NUM_CLASSES):
    """ResNet50，输出 num_classes 类（CIFAR-10 为 10）"""
    # 使用预定义结构，不加载 ImageNet 权重，从零训练
    model = resnet50(weights=None, num_classes=num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, epoch, total_epochs, steps_per_epoch):
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 50 == 0 or i == steps_per_epoch - 1:
            print(
                "Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.3f]"
                % (epoch + 1, total_epochs, i + 1, steps_per_epoch, loss.item())
            )
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    print("设备: %s" % DEVICE)
    if DEVICE.type == "cuda":
        print("GPU: %s" % torch.cuda.get_device_name(0))

    train_loader, test_loader = get_cifar10_loaders()
    steps_per_epoch = len(train_loader)

    model = build_resnet50().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs(BEST_CKPT_DIR, exist_ok=True)
    best_acc = 0.0

    print("Start Training Loop ...")
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            epoch, NUM_EPOCHS, steps_per_epoch
        )
        acc = evaluate(model, test_loader)
        print("-" * 50)
        print(
            "Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]"
            % (epoch + 1, NUM_EPOCHS, avg_loss, acc)
        )
        print("-" * 50)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": acc,
                "num_classes": NUM_CLASSES,
            }, BEST_CKPT_PATH)
            print("已保存最佳模型: %s" % BEST_CKPT_PATH)

    print("=" * 60)
    print("训练结束。最佳验证准确率: %.3f" % best_acc)
    print("最佳模型已保存至: %s" % BEST_CKPT_PATH)


if __name__ == "__main__":
    main()
