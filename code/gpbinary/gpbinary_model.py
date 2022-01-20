import numpy as np
import torch
from torch import nn

from prediction import pred_gp
from likelihood import negloglik_mvbinary
from fayans_support import get_Phi, get_psi, read_data, visualize_dataset


class MultiBinaryGP(nn.Module):
    def __init__(self, theta, x, y):
        super().__init__()
        self.theta = theta
        self.y = y
        self.psi = get_psi(y)
        self.Phi = get_Phi(x)
        self.d = theta.shape[1]
        self.m, self.n = y.shape
        self.kap = self.Phi.shape[1]

        # parameters
        self.lmb = nn.Parameter(torch.randn(self.d+1, self.kap), requires_grad=False)  # hyperparameter for kth GP, k=1, 2, ... kap
        self.sigma = nn.Parameter(torch.Tensor((1,)), requires_grad=False)  # noise parameter
        self.G = nn.Parameter(torch.randn(self.n, self.kap))  # principal component, k=1, 2, ... kap


    def forward(self, thetanew):
        lmb = self.lmb
        G = self.G
        sigma = self.sigma

        ypred = pred(lmb=lmb, G=G, sigma=sigma, thetanew=thetanew, theta=self.theta, psi=psi, Phi=Phi)
        return ypred

    def lik(self):
        lmb = self.lmb
        G = self.G
        sigma = self.sigma

        theta = self.theta
        y = self.y
        psi = self.psi
        Phi = self.Phi

        return negloglik_mvbinary(lmb=lmb, sigma=sigma, G=G, theta=theta, y=y, psi=psi, Phi=Phi)

    def accuracy(self, y, ypred):
        return (y == ypred).sum() / ypred.numel()


def pred(lmb, G, sigma, thetanew, theta, psi, Phi):
    kap = Phi.shape[1]
    n0 = thetanew.shape[0]

    # loop through kap dim of G
    G0 = torch.zeros(n0, kap)
    for k in range(kap):
        G0[:, k], _ = pred_gp(lmb=lmb[:, k], theta=theta, thetanew=thetanew, g=G[:, k])

    z0 = (psi + Phi @ G0.T) / sigma
    ypred = z0 > 0

    return ypred


if __name__ == '__main__':
    f0, x0, theta0 = read_data(r'../data/')
    y0 = np.isnan(f0).astype(int)

    f0 = torch.tensor(f0)
    x0 = torch.tensor(x0)
    theta0 = torch.tensor(theta0)
    y0 = torch.tensor(y0)

    # choose training and testing data
    failinds = np.argsort(y0.sum(0))
    traininds = failinds[-250:-50][::4]
    testinds = np.setdiff1d(failinds[-250:-50], traininds)

    ytr = y0[:, traininds]
    thetatr = theta0[traininds]
    yte = y0[:, testinds]
    thetate = theta0[testinds]

    psi = get_psi(ytr)
    Phi = get_Phi(x0)

    lr = 10e-3
    model = MultiBinaryGP(thetatr, x0, ytr)
    model.double()

    optim = torch.optim.Adam(model.parameters(), lr)

    header = ['iter', 'negloglik', 'accuracy']
    print('{:<5s} {:<12s} {:<10s}'.format(*header))
    for epoch in range(500):
        model.forward(thetatr)
        lik = model.lik()
        lik.backward()
        optim.step()
        if epoch % 50 == 0:
            print('{:<5d} {:<12.6f} {:<10.3f}'.format(epoch, lik, model.accuracy(yte, model(thetate))))
