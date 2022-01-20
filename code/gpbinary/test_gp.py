import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)

from prediction import pred_gp
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
        gpred, gpredvar = pred_gp(self.lmb, self.theta, theta0, self.g)
        return ((gpred - g0) ** 2).mean()


def test_gp():
    theta = torch.arange(0, 5, 1).resize_(5, 1)
    g = torch.normal(torch.sin(theta) + theta, 1)
    lmb = torch.Tensor((10, 0))

    thetatest = torch.arange(-1 / 3, 5, 1/6).resize_(32, 1)
    gtest = torch.sin(thetatest) + thetatest

    lr = 10e-1
    model = simpleGP(lmb, theta, g)
    model.double()
    model.requires_grad_()

    optim = torch.optim.LBFGS(model.parameters(), lr, line_search_fn='strong_wolfe')
    lik = model.lik()
    header = ['iter', 'negloglik', 'test mse']
    print('{:<5s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(10):
        optim.zero_grad()
        lik = model.lik()
        lik.backward()
        optim.step(lambda: model.lik())

        mse = model.test_mse(thetatest, gtest)
        print('{:<5d} {:<12.6f} {:<10.3f}'.format(epoch, lik, mse))


    gpred, gpredvar = pred_gp(model.lmb, model.theta, thetatest, model.g)

    import matplotlib.pyplot as plt
    thetatest = thetatest.numpy().squeeze()
    gpred = gpred.detach().numpy().squeeze()
    gpredvar = gpredvar.detach().numpy().squeeze()
    plt.plot(thetatest, gpred, 'k-')
    plt.fill_between(thetatest, gpred-2*gpredvar, gpred+2*gpredvar, alpha=0.75)
    plt.scatter(theta, g, marker='o', s=25, c='blue')
    plt.show()
    return


def test_gp_borehole():
    from test_function import gen_borehole_data
    theta, f, thetatest, ftest = gen_borehole_data()

    lmb = torch.Tensor(torch.zeros(9))

    lr = 10e-2
    model = simpleGP(lmb, theta, f)
    model.double()
    model.requires_grad_()

    optim = torch.optim.LBFGS(model.parameters(), lr, line_search_fn='strong_wolfe')
    header = ['iter', 'negloglik', 'test mse']
    print('{:<5s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(5):
        optim.zero_grad()
        lik = model.lik()
        lik.backward()
        optim.step(lambda: model.lik())

        mse = model.test_mse(thetatest, ftest)
        print('{:<5d} {:<12.6f} {:<10.3f}'.format(epoch, lik, mse))

    fpred, fpredvar = pred_gp(model.lmb, theta, thetatest, f)

    import numpy as np
    print('\ngp mse: {:.3f}'.format(np.mean((fpred.detach().numpy() - ftest.numpy()) ** 2)))

    from surmise.emulation import emulator
    x = np.array(1).reshape((1, 1))
    emu = emulator(x, theta.numpy(),
                   f.numpy().reshape(1, f.shape[0]),
                   method='PCGP',
                   args={'epsilon': 0.0})
    emupred = emu.predict(x=x, theta=thetatest.numpy())
    emumse = ((emupred.mean() - ftest.numpy()) ** 2).mean()

    print('\nsurmise mse: {:.3f}'.format(emumse))

    #
    # import matplotlib.pyplot as plt
    # thetatest = thetatest.numpy().squeeze()
    # fpred = fpred.detach().numpy().squeeze()
    # fpredvar = fpredvar.detach().numpy().squeeze()
    # plt.plot(thetatest[:, 0], fpred, 'k-')
    # plt.fill_between(thetatest[:, 0], fpred-2*fpredvar, fpred+2*fpredvar, alpha=0.25)
    # plt.scatter(theta[:, 0], f, marker='o', s=25, c='blue')
    # plt.xlabel(r'$\theta_0$')
    # plt.ylabel(r'$f$')
    # plt.show()


if __name__ == '__main__':
    test_gp()
    # test_gp_borehole()
