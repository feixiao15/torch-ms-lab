# -*- coding: utf-8 -*-
"""保存官方预训练 GoogLeNet 权重。"""
import os
import torch
from torchvision.models import googlenet, GoogLeNet_Weights

weights = GoogLeNet_Weights.IMAGENET1K_V1
model = googlenet(weights=weights)
out = os.path.join(os.path.dirname(__file__), "googlenet-1378be20.pth")
torch.save(model.state_dict(), out)
print("saved:", out)