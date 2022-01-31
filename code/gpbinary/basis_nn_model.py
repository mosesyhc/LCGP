import torch
from torch import nn
from gram_schmidt import gram_schmidt


class BasisGenNNTypeSingle(nn.Module):
    def __init__(self, x):
        super(BasisGenNNTypeSingle, self).__init__()
        d = x.shape[1]
        self.layer = nn.Sequential(
            nn.Linear(d, 3*d),
            nn.ReLU(),
            nn.Linear(3*d, 2*d),
            nn.ReLU(),
            nn.Linear(2*d, d),
            nn.LeakyReLU(),
            nn.Linear(d, 1)
        )

    def forward(self, x):
        return self.layer(x)


class BasisGenNNTypeMulti(nn.Module):
    def __init__(self, kap, x):
        super(BasisGenNNTypeMulti, self).__init__()
        self.x = x
        self.m, self.d = x.shape
        self.kap = kap
        self.models = nn.ModuleList([BasisGenNNTypeSingle(x) for k in range(kap)])

    def forward(self, x):
        Phi = torch.zeros(self.m, self.kap)
        for k, model in enumerate(self.models):
            Phi[:, k] = model(x).squeeze()
        orthoPhi = gram_schmidt(Phi)
        return orthoPhi


class BasisGenNNType_OLD(nn.Module):
    def __init__(self, kap):
        super(BasisGenNNType_OLD, self).__init__()
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
