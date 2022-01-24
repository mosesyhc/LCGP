import torch
from torch import nn


class BasisGenNN(nn.Module):
    def __init__(self, n, kap):
        super(BasisGenNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 200),
            nn.LeakyReLU(),
            nn.Linear(200, n * kap),
        )

    def forward(self, x):
        return self.layer(x)
