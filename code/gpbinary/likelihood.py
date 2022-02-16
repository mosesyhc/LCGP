import torch
import torch.distributions.normal as Normal
from matern_covmat import covmat
norm = Normal.Normal(loc=0, scale=1)


# named inputs in function calls
def negloglik_mvbinary(lmb, G, theta, y, psi, Phi):
    # hyperparameter organization (size 2d + 2 + 1 + kap*n), kap = 2:
    # hyp = (lambda_1, lambda_2, sigma, G_11, G_21, ..., G_n1, G_12, ..., Gn2)
    # (lambda_k1, ..., lambda_kd) are the lengthscales for theta, k = 1, 2
    # lambda_k(d+1) is the scale for GP, k = 1, 2
    # sigma is the noise parameter in the indicator function

    kap = Phi.shape[1]

    nll = negloglik_link(G, y, psi, Phi)
    for k in range(kap):
        nll += negloglik_gp(lmb[:, k], theta, G[:, k])
    return nll


def negloglik_link(G, y, psi, Phi):
    z = (psi + Phi @ G.T)
    F = norm.cdf(z)
    ypos = y > 0.5
    negloglik = -(torch.log(F[ypos])).sum()  # ones
    negloglik -= (torch.log((1 - F[~ypos]))).sum()  # zeros

    return negloglik


def negloglik_gp(lmb, theta, g, lmbregmean=0, lmbregstd=1):
    R = covmat(theta, theta, lmb)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(W)
    fcenter = Vh.T @ g
    n = g.shape[0]

    sig2hat = (n * torch.mean(fcenter ** 2) + 10) / (n + 10)
    negloglik = 1/2 * torch.sum(torch.log(W))  # log-determinant
    negloglik += n/2 * torch.log(sig2hat)  # log of MLE of scale
    negloglik += 1/2 * torch.sum(fcenter ** 2 / sig2hat)  # quadratic term
    negloglik += 1/2 * torch.sum(((lmb - lmbregmean + 10e-8) / lmbregstd)**2)  # regularization of hyperparameter

    return negloglik, Vh
