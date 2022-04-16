import torch
from torch import nn
from gram_schmidt import gram_schmidt


class Basis(nn.Module):
    def __init__(self, m, kap, normalize=True, inputdata=None):
        super(Basis, self).__init__()
        if inputdata is not None:
            self.Phi = nn.Parameter(torch.linalg.svd(inputdata)[0][:, :kap])
        else:
            self.Phi = nn.Parameter(torch.eye(m)[:, :kap])
        self.normalize = normalize

    def forward(self):
        if self.normalize:
            return gram_schmidt(self.Phi)
        else:
            return self.Phi


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
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

    def forward(self, x):
        return self.layer(x)


class BasisGenNNTypeMulti(nn.Module):
    def __init__(self, kap, x, normalize=True):
        super(BasisGenNNTypeMulti, self).__init__()
        self.x = x
        self.m, self.d = x.shape
        self.kap = kap
        self.models = nn.ModuleList([BasisGenNNTypeSingle(x) for k in range(kap)])
        self.gs = normalize

    def forward(self, x):
        Phi = torch.cat([model(x) for model in self.models], dim=1)
        # Phi = torch.zeros(self.m, self.kap)
        # for k, model in enumerate(self.models):
        #     Phi[:, k] = model(x).squeeze()
        if self.gs:
            Phi = gram_schmidt(Phi)

        return Phi


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


def init_weights(m):
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight) #, mean=0.0, std=0.1) ## or simply use your layer.reset_parameters()
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight) #, mean=0.0, std=torch.sqrt(1 / m.in_features))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight) #, mean=0.0, std=np.sqrt(4 / m.in_channels))
        if m.bias is not None:
            nn.init.zeros_(m.bias)