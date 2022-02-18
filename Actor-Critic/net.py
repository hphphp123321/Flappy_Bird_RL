from cmath import tanh
from turtle import forward
import numpy as np
import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class PiNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, out_dim)
        )
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    x = torch.tensor([209., 132.,   1.])
    model = PiNet(in_dim=3, out_dim=2)
    # model.to(device='cuda:0')
    y = model(x)
    print(y)
    p = torch.rand(1)
    if p < y[0]:
        print("yes")
    else:
        print("no")