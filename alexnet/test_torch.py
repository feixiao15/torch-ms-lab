import torch
import numpy as np
from torchvision.models import alexnet, AlexNet_Weights

# 1. 固定随机种子（非常重要）
torch.manual_seed(0)
np.random.seed(0)

# 2. 加载官方预训练 AlexNet
weights = AlexNet_Weights.IMAGENET1K_V1
model = alexnet(weights=weights)
model.eval()

# 3. 构造固定输入（不要用随机每次都变）
x_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
x_pt = torch.from_numpy(x_np)

# 4. 前向推理
with torch.no_grad():
    logits_pt = model(x_pt).numpy()

print(logits_pt.shape)  # (1, 1000)
