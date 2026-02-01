# -*- coding: utf-8 -*-
"""
将 PyTorch AlexNet checkpoint 转为 MindSpore checkpoint。
AlexNet 无 BN，参数名一一对应。
用法: python alexnet/pth_to_ckpt.py
"""
import os
import sys
import torch
import mindspore as ms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PTH = os.path.join(ROOT, "alexnet", "checkpoints", "alexnet-owt-7be5be79.pth")
CKPT = os.path.join(ROOT, "alexnet", "checkpoints", "alexnet.ckpt")


def convert():
    from alexnet.ms_alexnet import AlexNet

    raw = torch.load(PTH, map_location="cpu", weights_only=True)
    state = raw if not (isinstance(raw, dict) and "state_dict" in raw) else raw["state_dict"]
    state = {k: v for k, v in state.items() if "num_batches_tracked" not in k}

    ms_net = AlexNet(num_classes=1000)
    ms_params = {p.name: p for p in ms_net.get_parameters()}
    new_params = []
    for ms_name, ms_param in ms_params.items():
        if ms_name in state:
            pt_val = state[ms_name]
            if tuple(pt_val.shape) == tuple(ms_param.shape):
                new_params.append({"name": ms_name, "data": ms.Tensor(pt_val.detach().numpy().astype("float32"))})
            else:
                print("shape mismatch:", ms_name, pt_val.shape, "vs", ms_param.shape)
        else:
            print("no pt param:", ms_name)

    ms.save_checkpoint(new_params, CKPT)
    print("已保存 %d 个参数到 %s" % (len(new_params), CKPT))


if __name__ == "__main__":
    convert()
