import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import os

weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights)
out = os.path.join(os.path.dirname(__file__), "mobilenet_v2-b0353104.pth")
torch.save(model.state_dict(), out)
print("saved:", out)