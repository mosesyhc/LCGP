import torch
from matern_covmat import covmat


def negloglik(hyp, theta, g, hypregmean, hypregstd):
    R = covmat(theta, theta, hyp)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(torch.abs(W))
    fcenter = Vh.T @ g
    n = g.shape[0]

    sig2hat = (n * torch.mean(fcenter ** 2) + 1) / (n + 1)
    negloglik0 = 1/2 * torch.sum(torch.log(torch.abs(W))) + n/2 * torch.log(sig2hat)
    negloglik0 += 1/2 * torch.sum(((hyp - hypregmean)/hypregstd)**2)

    return negloglik0


def negloglikgrad(hyp, theta, g, hypregmean, hypregstd):
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

    dnegloglik += (10 ** (-8) + hyp - hypregmean) / (hypregstd ** 2)

    return dnegloglik
