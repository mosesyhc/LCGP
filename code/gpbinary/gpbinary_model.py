import numpy as np
import torch
import torch.distributions.normal as Normal
from likelihood import negloglik, negloglikgrad
from matern_covmat import covmat
torch.set_default_dtype(torch.float64)
norm = Normal.Normal(0, 1)


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
    R = covmat(theta, theta, lmb)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(torch.abs(W))

    Rinv_g = Vh @ Vh.T @ g
    Rnew = covmat(thetanew, theta, lmb)

    return Rnew @ Rinv_g


def pred(hyp, thetanew, theta, psi, Phi):
    d = theta.shape[1]
    kap = Phi.shape[1]
    n0 = thetanew.shape[0]

    G = hyp[-(kap*n):]
    G.resize_(kap, n).transpose_(0, 1)  # G is n x kap
    lmb1 = hyp[:(d+1)]
    lmb2 = hyp[(d+1):(2*d+2)]
    sigma = hyp[2*d+2]

    G0 = torch.zeros(n0, kap)
    G0[:, 0] = pred_gp(lmb1, theta, thetanew, G[:, 0])
    G0[:, 1] = pred_gp(lmb2, theta, thetanew, G[:, 1])

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
    d = 13
    m, n = ytr.shape
    kap = Phi.shape[1]

    # hyp = torch.ones(2*d + 2 + 1 + kap*n)
    # nll = negloglik(hyp, torch.tensor(thetatr), torch.tensor(ytr), psi, Phi)
    # dnll = negloglikgrad(hyp, torch.tensor(thetatr), torch.tensor(ytr), psi, Phi)
    # print(nll)
    # print(dnll)

    ## hyperparameter organization (size 2d + 2 + 1 + kap*n), kap = 2:
    ## hyp = (lambda_1, lambda_2, sigma, G_11, G_21, ..., G_n1, G_12, ..., Gn2)
    ## (lambda_k1, ..., lambda_kd) are the lengthscales for theta, k = 1, 2
    ## lambda_k(d+1) is the scale for GP, k = 1, 2
    ## sigma is the noise parameter in the indicator function
    hyp0 = torch.zeros(2*d + 2 + 1 + kap*n)
    hyp0[:d] = 0 + 0.5 * torch.log(torch.tensor(d)) + torch.log(torch.std(thetatr, dim=0))
    hyp0[d] = 0
    hyp0[(d+1):(2*d + 1)] = 0 + 0.5 * torch.log(torch.tensor(d)) + torch.log(torch.std(thetatr, dim=0))
    hyp0[2*d+1] = 0
    hyp0[2*d+2] = 2

    alpha = 10e-5
    nll = negloglik(hyp0, thetatr, ytr, psi, Phi)
    dnll = negloglikgrad(hyp0, thetatr, ytr, psi, Phi)
    ypred = pred(hyp0, thetate, thetatr, psi, Phi)

    header = ['iter', 'negloglik', 'dnegloglik', 'accuracy']
    print('{:<5s} {:<12s} {:<12s} {:<10s}'.format(*header))
    for i in range(50):
        print('{:<5d} {:<12.6f} {:<12.6f} {:<10.3f}'.format(i, nll, dnll.mean(), (ypred == yte).sum() / yte.numel()))
        hyp0 -= alpha * dnll
        nll = negloglik(hyp0, thetatr, ytr, psi, Phi)
        dnll = negloglikgrad(hyp0, thetatr, ytr, psi, Phi)
        ypred = pred(hyp0, thetate, thetatr, psi, Phi)

        if i == 10:
            pass
