import numpy as np
import torch
import torch.distributions.normal as Normal
from torch import nn
from likelihood import negloglik
from matern_covmat import covmat
torch.set_default_dtype(torch.float64)
norm = Normal.Normal(0, 1)


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

        return negloglik(lmb=lmb, sigma=sigma, G=G, theta=theta, y=y, psi=psi, Phi=Phi)

    def accuracy(self, y, ypred):
        return (y == ypred).sum() / ypred.numel()


def read_data(dir):
    f = np.loadtxt(dir + r'f.txt')
    x = np.loadtxt(dir + r'x.txt')
    theta = np.loadtxt(dir + r'theta.txt')
    return f, x, theta


def visualize_dataset(ytrain, ytest):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(-ytrain.T, aspect='auto', cmap='gray', interpolation='none')
    ax[1].imshow(-ytest.T, aspect='auto', cmap='gray', interpolation='none')
    ax[0].set_title('Training data')
    ax[0].set_ylabel('Parameters')
    ax[1].set_title('Testing data')
    plt.show()


def get_psi(y):
    # y = self.y
    z = (y.sum(1) + 10) / (y.shape[1] + 20)
    psi = norm.icdf(z)
    return psi.unsqueeze(1)  # returns m x 1


def get_Phi(x):
    # x = self.x
    tmp = x[:, :2]
    tmp[:, 0] -= tmp[:, 1]  # Use (N, Z) instead of (A, Z)
    Phi = (tmp - tmp.mean(0)) / tmp.std(0)
    return Phi  # returns m x kappa


def pred_gp(lmb, theta, thetanew, g):
    '''
    Test in test_gp.py.

    :param lmb: hyperparameter for the covariance matrix
    :param theta: set of training parameters (size n x d)
    :param thetanew: set of testing parameters (size n0 x d)
    :param g: reduced rank latent variables (size n x 1)
    :return:
    '''

    # covariance matrix R for the training thetas
    R = covmat(theta, theta, lmb)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(torch.abs(W))  # check abs?

    Rinv_g = Vh @ Vh.T @ g
    Rnewold = covmat(thetanew, theta, lmb)
    Rnewnew = covmat(thetanew, thetanew, lmb)

    predmean = Rnewold @ Rinv_g
    predvar = Rnewnew - Rnewold @ Vh @ Vh.T @ Rnewold.T
    return predmean, predvar.diag()


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
