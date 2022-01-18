import torch
from torch import nn

import matern_covmat
from gpbinary_model import pred_gp
import torch.distributions.multivariate_normal as MVN
import matplotlib.pyplot as plt
from likelihood import negloglik_gp


class simpleGP(nn.Module):
    def __init__(self, lmb, theta, g):
        super().__init__()
        self.lmb = nn.Parameter(lmb)
        self.theta = theta
        self.g = g

    def forward(self, theta0):
        lmb = self.lmb
        theta = self.theta
        g = self.g

        pred, predvar = pred_gp(lmb, theta, theta0, g)
        return pred

    def lik(self):
        lmb = self.lmb
        theta = self.theta
        g = self.g
        return negloglik_gp(lmb, theta, g)

    def test_mse(self, theta0, g0):
        gpred, sigma2pred = pred_gp(self.lmb, self.theta, theta0, self.g)
        return ((gpred - g0)**2).mean()


def test_gp():
    theta = torch.arange(0, 5, 1).resize_(5, 1)
    g = torch.normal(torch.sin(theta) + theta, 1)
    lmb = torch.Tensor((-2, 1))

    thetatest = torch.arange(1/3, 5, 0.25).resize_(19, 1)
    gtest = torch.sin(thetatest) + thetatest

    lr = 10e-2
    model = simpleGP(lmb, theta, g)

    optim = torch.optim.Adam(model.parameters(), lr)
    header = ['iter', 'negloglik', 'test mse']
    print('{:<5s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(20):
        model.forward(theta)
        lik = model.lik()
        lik.backward()
        optim.step()
        # print(model.lmb, model.lik(), model.lmb.grad)
        # if epoch % 5 == 0:
        print('{:<5d} {:<12.6f} {:<10.3f}'.format(epoch, lik, model.test_mse(model(thetatest), gtest)))

    return


def test_gp_borehole():
    from test_function import gen_borehole_data
    theta, f, thetatest, ftest = gen_borehole_data()

    lmb = torch.Tensor((-2, -2, -2, -2, -2, -2, -2, -2, 1))

    lr = 10e-4
    model = simpleGP(lmb, theta, f)

    optim = torch.optim.LBFGS(model.parameters())
    header = ['iter', 'negloglik', 'test mse']
    print('{:<5s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(20):
        optim.zero_grad()
        model.forward(theta)
        lik = model.lik()
        lik.backward()
        optim.step(lambda: model.lik())

        mse = model.test_mse(thetatest, ftest)
        # print(model.lmb, model.lik(), model.lmb.grad)
        # if epoch % 5 == 0:
        print('{:<5d} {:<12.6f} {:<10.3f}'.format(epoch, lik, mse))


    import numpy as np
    from surmise.emulation import emulator
    x = np.array(1).reshape((1, 1))
    emu = emulator(x, theta.numpy(), f.numpy().reshape(1, f.shape[0]))
    emupred = emu.predict(x, thetatest.numpy()).mean()
    emumse = ((emupred - ftest.numpy())**2).mean()


    print('\nsurmise mse: {:.3f}'.format(emumse))


if __name__ == '__main__':
    # test_gp()
    test_gp_borehole()