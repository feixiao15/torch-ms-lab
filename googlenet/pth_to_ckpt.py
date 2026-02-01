# -*- coding: utf-8 -*-
"""
将 PyTorch GoogLeNet checkpoint 转为 MindSpore checkpoint。
BN 映射: gamma/beta/moving_mean/moving_variance <-> weight/bias/running_mean/running_var
跳过 aux1、aux2 参数（仅主 logits 推理）。
"""
import os
import sys
import torch
import mindspore as ms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PTH = os.path.join(ROOT, "googlenet", "googlenet-1378be20.pth")
CKPT = os.path.join(ROOT, "googlenet", "googlenet.ckpt")

BN_MAP = {"weight": "gamma", "bias": "beta", "running_mean": "moving_mean", "running_var": "moving_variance"}


def pt_to_ms_key(pt_key):
    if "num_batches_tracked" in pt_key or "aux1" in pt_key or "aux2" in pt_key:
        return None
    parts = pt_key.split(".")
    if parts[-1] in BN_MAP and ".bn" in pt_key:
        ms_suffix = BN_MAP[parts[-1]]
        return ".".join(parts[:-1] + [ms_suffix])
    return pt_key


def convert():
    from googlenet.ms_googlenet import GoogLeNet

    raw = torch.load(PTH, map_location="cpu", weights_only=True)
    state = {k: v for k, v in raw.items() if "num_batches_tracked" not in k and "aux1" not in k and "aux2" not in k}

    ms_net = GoogLeNet(num_classes=1000)
    ms_params = {p.name: p for p in ms_net.get_parameters()}
    new_params = []
    for pt_key, pt_val in state.items():
        ms_key = pt_to_ms_key(pt_key)
        if ms_key is None:
            continue
        if ms_key in ms_params:
            pt_val = pt_val.detach().numpy().astype("float32")
            if pt_val.shape == tuple(ms_params[ms_key].shape):
                new_params.append({"name": ms_key, "data": ms.Tensor(pt_val)})
            else:
                print("shape mismatch:", pt_key, "->", ms_key, pt_val.shape, ms_params[ms_key].shape)
        else:
            print("no ms param:", pt_key, "->", ms_key)

    ms.save_checkpoint(new_params, CKPT)
    print("已保存 %d 个参数到 %s" % (len(new_params), CKPT))


if __name__ == "__main__":
    convert()
