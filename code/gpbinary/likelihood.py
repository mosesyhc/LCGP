import torch
import torch.distributions.normal as Normal
from matern_covmat import covmat
norm = Normal.Normal(loc=0, scale=1)


def negloglik(hyp, theta, y, psi, Phi):
    # hyperparameter organization:
    # hyp = (lambda_1, lambda_2, sigma, G_11, G_21, ..., G_n1, G_12, ..., Gn2)
    # (lambda_k1, ..., lambda_kd) are the lengthscales for theta, k = 1, 2
    # lambda_k(d+1) is the scale for GP, k = 1, 2
    # sigma is the noise parameter in the indicator function

    d = theta.shape[1]
    m, n = y.shape
    kap = Phi.shape[1]

    G = hyp[-(kap*n):]
    G.resize_(kap, n).transpose_(0, 1)  # G is n x kap
    lmb1 = hyp[:(d+1)]
    lmb2 = hyp[(d+1):(2*d+2)]
    sigma = hyp[2*d+2]

    nll = negloglik_link(sigma, y, psi, Phi, G) + \
        negloglik_gp(lmb1, theta, G[:, 0]) + \
        negloglik_gp(lmb2, theta, G[:, 1])

    return nll


def negloglikgrad(hyp, theta, y, psi, Phi):
    dnegloglik = torch.zeros(hyp.shape[0])

    d = theta.shape[1]
    m, n = y.shape
    kap = Phi.shape[1]

    G = hyp[-(kap*n):]
    G.resize_(kap, n).transpose_(0, 1)  # G is n x kap
    lmb1 = hyp[:(d+1)]
    lmb2 = hyp[(d+1):(2*d+2)]
    sigma = hyp[2*d+2]

    dnegloglik[:(d+1)], R1inv = negloglikgrad_gp(lmb1, theta, G[:, 0])
    dnegloglik[(d+1):(2*d+2)], R2inv = negloglikgrad_gp(lmb2, theta, G[:, 1])
    dnegloglik[2*d+2] = negloglikgrad_link(sigma, y, psi, Phi, G)
    dnegloglik[-(2*n):-n] = R1inv @ G[:, 0]
    dnegloglik[-n:] = R2inv @ G[:, 1]
    return dnegloglik


def negloglik_link(sigma, y, psi, Phi, G):
    z = (psi + Phi @ G.T) / sigma
    F = norm.cdf(z)
    ypos = y > 0.5
    negloglik = -(torch.log(F[ypos])).sum()  # ones
    negloglik -= (torch.log((1 - F[~ypos]))).sum()  # zeros

    return negloglik


def negloglikgrad_link(sigma, y, psi, Phi, G):
    z = (psi + Phi @ G.T) / sigma
    F = norm.cdf(z)
    f = torch.exp(norm.log_prob(z))
    ypos = y > 0.5

    dnegloglik = (z / sigma * f[ypos] / F[ypos]).sum()  # ones
    dnegloglik -= ((z/ sigma) * f[ypos] / (1 - F[ypos])).sum()  # zeros
    return dnegloglik


def negloglik_gp(lmb, theta, g, lmbregmean=None, lmbregstd=None):
    R = covmat(theta, theta, lmb)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(torch.abs(W))
    fcenter = Vh.T @ g
    n = g.shape[0]

    sig2hat = (n * torch.mean(fcenter ** 2) + 1) / (n + 1)
    negloglik = 1/2 * torch.sum(torch.log(torch.abs(W))) + n/2 * torch.log(sig2hat)
    # negloglik += 1/2 * torch.sum(((lmb - lmbregmean)/lmbregstd)**2)

    return negloglik


def negloglikgrad_gp(lmb, theta, g, lmbregmean=None, lmbregstd=None):
    R, dR = covmat(theta, theta, lmb, return_gradhyp=True)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(torch.abs(W))
    fcenter = (Vh.T @ g).unsqueeze(1)
    n = g.shape[0]

    sig2hat = (n * torch.mean(fcenter ** 2) + 1) / (n + 1)
    outerprod = Vh @ fcenter @ fcenter.T @ Vh.T
    Rinv = Vh @ Vh.T

    dnegloglik = torch.zeros(dR.shape[2])
    for j in range(dR.shape[2]):
        dsig2hat = - torch.sum(outerprod * dR[:, :, j]) / (n + 1)
        dnegloglik[j] += 1/2 * n * dsig2hat / sig2hat
        dnegloglik[j] += 1/2 * torch.sum(Rinv * dR[:, :, j])

    # dnegloglik += (10 ** (-8) + lmb - lmbregmean) / (lmbregstd ** 2)
    return dnegloglik, Rinv
