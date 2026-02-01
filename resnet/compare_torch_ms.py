# -*- coding: utf-8 -*-
"""
按 MindSpore 迁移指南对比 PyTorch 与 MindSpore ResNet50：
1) pth 转 ckpt（BN 参数映射）
2) 比较准确度
3) 比较同一输入的 logits 误差
参考: https://www.mindspore.cn/docs/zh-CN/r2.0/migration_guide/sample_code.html
在 cmd 中运行: python compare_torch_ms.py
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torchvision
import torchvision.transforms as trans
import mindspore as ms
from mindspore import nn

from resnet_pytorch.resnet import resnet50 as pt_resnet50
from resnet_ms.src.resnet import resnet50 as ms_resnet50

# ============ 路径（注意 resnet_pytroch_res 拼写） ============
PTH_PATH = os.path.join(ROOT, "resnet_pytroch_res", "resnet.pth")
CKPT_PATH = os.path.join(ROOT, "resnet_pytroch_res", "resnet50.ckpt")
DATA_ROOT = os.path.join(ROOT, "data")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_BATCHES = 100  # None=全量; 设为 10 可快速验证（约 320 样本）


def param_convert(pt_params, ckpt_path):
    """按迁移指南将 PyTorch 参数转为 MindSpore checkpoint
    BN 映射: MS gamma/beta/moving_mean/moving_variance <-> PT weight/bias/running_mean/running_var
    """
    ms2pt_suffix = {"gamma": "weight", "beta": "bias", "moving_mean": "running_mean", "moving_variance": "running_var"}
    ms_net = ms_resnet50(num_classes=10)
    ms_params = {p.name: p.data.shape for p in ms_net.get_parameters()}
    new_params_list = []
    for ms_name, ms_shape in ms_params.items():
        if ("bn" in ms_name or "downsample.1" in ms_name) and ms_name.split(".")[-1] in ms2pt_suffix:
            parts = ms_name.split(".")
            pt_suffix = ms2pt_suffix[parts[-1]]
            pt_name = ".".join(parts[:-1] + [pt_suffix])
        else:
            pt_name = ms_name
        if pt_name in pt_params and pt_params[pt_name].shape == ms_shape:
            val = pt_params[pt_name].astype(np.float32)
            new_params_list.append({"name": ms_name, "data": ms.Tensor(val)})
        else:
            print("[转换] 未匹配:", ms_name, "->", pt_name)
    ms.save_checkpoint(new_params_list, ckpt_path)
    print("已保存 MindSpore checkpoint:", ckpt_path)


def load_pt_params(pth_path):
    """加载 PyTorch 参数，跳过 num_batches_tracked"""
    try:
        raw = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(pth_path, map_location="cpu")
    raw = raw.get("model_state_dict", raw)
    return {k: v.numpy() for k, v in raw.items() if "num_batches_tracked" not in k}


def get_test_loader():
    """CIFAR10 测试集，与迁移指南一致：Resize(224), Normalize"""
    tf = trans.Compose([
        trans.Resize(IMAGE_SIZE),
        trans.ToTensor(),
        trans.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


def main():
    ms.set_context(device_target="CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(PTH_PATH):
        print("未找到:", PTH_PATH)
        return
    pt_params = load_pt_params(PTH_PATH)
    param_convert(pt_params, CKPT_PATH)

    pt_net = pt_resnet50(num_classes=10).to(device)
    try:
        raw_pt = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        raw_pt = torch.load(PTH_PATH, map_location="cpu")
    pt_state = raw_pt.get("model_state_dict", raw_pt)
    pt_net.load_state_dict(pt_state, strict=False)
    pt_net.eval()

    ms_net = ms_resnet50(num_classes=10).set_train(False)
    ms.load_checkpoint(CKPT_PATH, ms_net)

    loader = get_test_loader()

    # 准确度
    pt_correct, ms_correct, total = 0, 0, 0
    pt_logits_all, ms_logits_all, labels_all = [], [], []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
                break
            data_np = data.numpy().astype(np.float32)
            data_dev = data.to(device)
            pt_out = pt_net(data_dev)
            pt_pred = pt_out.argmax(dim=1)
            pt_correct += (pt_pred == target.to(device)).sum().item()

            ms_out = ms_net(ms.Tensor(data_np))
            ms_out_np = ms_out.asnumpy()
            ms_pred = ms_out_np.argmax(axis=1)
            ms_correct += (ms_pred == target.numpy()).sum()

            total += target.size(0)
            pt_logits_all.append(pt_out.cpu().numpy())
            ms_logits_all.append(ms_out_np)
            labels_all.append(target.numpy())

    pt_acc = 100.0 * pt_correct / total
    ms_acc = 100.0 * ms_correct / total
    print("\n========== 准确度 ==========")
    print("PyTorch 测试集准确率: %.2f%%" % pt_acc)
    print("MindSpore 测试集准确率: %.2f%%" % ms_acc)

    # Logits 误差（同一输入）
    pt_logits = np.concatenate(pt_logits_all, axis=0)
    ms_logits = np.concatenate(ms_logits_all, axis=0)
    diff = np.abs(pt_logits - ms_logits)
    max_err = diff.max()
    mean_err = diff.mean()
    print("\n========== Torch vs MindSpore logits 差值（同一输入） ==========")
    print("logits 最大绝对误差: %.6e" % max_err)
    print("logits 平均绝对误差: %.6e" % mean_err)

    print("\n========== 结果汇总 ==========")
    print("  PyTorch 准确率: %.2f%%" % pt_acc)
    print("  MindSpore 准确率: %.2f%%" % ms_acc)
    print("  logits 最大误差: %.6e" % max_err)
    print("  logits 平均误差: %.6e" % mean_err)


if __name__ == "__main__":
    main()
