import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnext101_32x8d

net = resnext101_32x8d()
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)