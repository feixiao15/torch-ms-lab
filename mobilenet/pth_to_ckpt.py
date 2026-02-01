# -*- coding: utf-8 -*-
"""PyTorch MobileNetV2 pth 权重转换为 MindSpore ckpt。"""
import os
import sys
import torch
import mindspore as ms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mobilenet.ms_mobilenetv2 import MobileNetV2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTH = os.path.join(ROOT, "mobilenet", "mobilenet_v2-b0353104.pth")
CKPT = os.path.join(ROOT, "mobilenet", "mobilenet_v2.ckpt")

# BatchNorm 参数映射：PyTorch -> MindSpore
BN_MAP = {
    "weight": "gamma",
    "bias": "beta",
    "running_mean": "moving_mean",
    "running_var": "moving_variance",
}


def convert():
    raw = torch.load(PTH, map_location="cpu", weights_only=True)
    state = raw if not (isinstance(raw, dict) and "state_dict" in raw) else raw["state_dict"]
    state = {k: v for k, v in state.items() if "num_batches_tracked" not in k}

    ms_net = MobileNetV2(num_classes=1000)
    ms_params = {p.name: p for p in ms_net.get_parameters()}

    new_params = []
    matched, skipped = 0, 0

    for pt_name, pt_val in state.items():
        ms_name = pt_name
        # BatchNorm 参数映射：先尝试 BN 后缀，若 MS 中无对应参数则保持原名（Conv）
        for pt_suffix, ms_suffix in BN_MAP.items():
            if pt_name.endswith("." + pt_suffix):
                candidate = pt_name.rsplit(".", 1)[0] + "." + ms_suffix
                if candidate in ms_params:
                    ms_name = candidate
                break

        if ms_name in ms_params:
            if tuple(pt_val.shape) == tuple(ms_params[ms_name].shape):
                new_params.append({
                    "name": ms_name,
                    "data": ms.Tensor(pt_val.numpy().astype("float32"))
                })
                matched += 1
            else:
                print("shape mismatch:", ms_name, pt_val.shape, "vs", ms_params[ms_name].shape)
                skipped += 1
        else:
            print("no ms param:", pt_name, "->", ms_name)
            skipped += 1

    ms.save_checkpoint(new_params, CKPT)
    print("转换完成: %d 个参数匹配, %d 个跳过" % (matched, skipped))
    print("保存到:", CKPT)


if __name__ == "__main__":
    convert()
