import torch
import torch.distributions.normal as Normal
from matern_covmat import covmat
norm = Normal.Normal(loc=0, scale=1)


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


def negloglik_gp(hyp, theta, g, hypregmean=None, hypregstd=None):
    R = covmat(theta, theta, hyp)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(torch.abs(W))
    fcenter = Vh.T @ g
    n = g.shape[0]

    sig2hat = (n * torch.mean(fcenter ** 2) + 1) / (n + 1)
    negloglik = 1/2 * torch.sum(torch.log(torch.abs(W))) + n/2 * torch.log(sig2hat)
    # negloglik += 1/2 * torch.sum(((hyp - hypregmean)/hypregstd)**2)

    return negloglik


def negloglikgrad_gp(hyp, theta, g, hypregmean=None, hypregstd=None):
    R, dR = covmat(theta, theta, hyp, return_gradhyp=True)

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

    # dnegloglik += (10 ** (-8) + hyp - hypregmean) / (hypregstd ** 2)
    return dnegloglik
