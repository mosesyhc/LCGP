import torch
from torch import nn


class BasisGenNNType(nn.Module):
    def __init__(self, kap):
        super(BasisGenNNType, self).__init__()
        self.kap = kap
        self.layer = nn.Sequential(
            nn.Linear(11, 25 * kap),
            nn.ReLU(),
            nn.Linear(25 * kap, 10 * kap),
            nn.ReLU(),
            nn.Linear(10 * kap, 5 * kap),
            nn.ReLU(),
            nn.Linear(5 * kap, 2 * kap),
            nn.LeakyReLU(),
            nn.Linear(2 * kap, kap),
        )

    def forward(self, x):
        return self.layer(x)


class BasisGenNN(nn.Module):
    def __init__(self, kap):
        super(BasisGenNN, self).__init__()
        self.kap = kap
        self.layer = nn.Sequential(
            nn.Linear(3, 20 * kap),
            nn.ReLU(),
            nn.Linear(20 * kap, 10 * kap),
            nn.ReLU(),
            nn.Linear(10 * kap, 5 * kap),
            nn.LeakyReLU(),
            nn.Linear(5 * kap, kap),
        )

    def forward(self, x):
        return self.layer(x)
