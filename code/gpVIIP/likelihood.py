import torch
import torch.distributions.normal as Normal
from matern_covmat import cormat, cov_sp
norm = Normal.Normal(loc=0, scale=1)


def negloglik_gp(llmb, lnug, theta, g):
    C0 = cormat(theta, theta, llmb)
    nug = torch.exp(lnug) / (1 + torch.exp(lnug))

    C = (1 - nug) * C0 + nug * torch.eye(theta.shape[0])

    W, V = torch.linalg.eigh(C)
    Vh = V / torch.sqrt(W.abs())
    fcenter = Vh.T @ g
    n = g.shape[0]

    tau2hat = (n * torch.mean(fcenter ** 2) + 10) / (n + 10)
    # print(tau2hat)

    negloglik = 1/2 * torch.sum(torch.log(W.abs()))  # log-determinant
    negloglik += n/2 * torch.log(tau2hat)  # log of MLE of scale

    # llmb, lsigma2 regularization
    llmbreg = 10 * (llmb + 1) ** 2
    llmbreg[-1] = 15 * llmb[-1] ** 2

    negloglik += llmbreg.sum() #+ 5 * (lsigma2 + 10)**2

    return negloglik


def negloglik_singlevar_gp(llmb, lsigma2, theta, g):
    C0 = cormat(theta, theta, llmb)
    nug = torch.exp(lsigma2) / (1 + torch.exp(lsigma2))
    C = (1 - nug) * C0 + nug * torch.eye(theta.shape[0])  # (1 - nug) *

    W, V = torch.linalg.eigh(C)
    Vh = V / torch.sqrt(W.abs())
    fcenter = Vh.T @ g
    n = g.shape[0]

    tau2hat = (n * torch.mean(fcenter ** 2) + 10) / (n + 10)
    # print(tau2hat)

    negloglik = 1/2 * torch.sum(torch.log(W.abs()))  # log-determinant
    negloglik += n/2 * torch.log(tau2hat)  # log of MLE of scale
    # negloglik += 1/2 * torch.sum(fcenter ** 2 / sig2hat)  # quadratic term

    # llmb, lsigma2 regularization
    llmbreg = 5 * (llmb + 1) ** 2
    llmbreg[-1] = 15 * llmb[-1] ** 2
    negloglik += llmbreg.sum() + 5 * (lsigma2 + 10)**2
    # negloglik += 1/2 * (llmb**2).sum() + 1/10 * (lsigma2 + 10)**2

    return negloglik


def negloglik_gp_sp(llmb, lnug, theta, thetai, g):
    Delta_inv_diag, _, _, QRinvh, logdet_C, _, _ = cov_sp(theta, thetai, llmb, lnug)

    n = g.shape[0]
    QRinvh_g = (QRinvh.T * g).sum(1)
    quad = g @ (Delta_inv_diag * g) - (QRinvh_g ** 2).sum()
    tau2hat = (quad + 10) / (n + 10)
    # print(tau2hat)

    negloglik = 1/2 * logdet_C  # log-determinant
    negloglik += n/2 * torch.log(tau2hat)  # log of MLE of scale
    # negloglik += 1/2 * quad / sig2hat  # quadratic term
    # negloglik += 1/2 * torch.sum(((llmb - llmbregmean) / llmbregstd)**2)  # regularization of hyperparameter

    # llmb, lsigma2 regularization
    llmbreg = 15 * (llmb + 1) ** 2
    llmbreg[-1] = 25 * llmb[-1] ** 2
    negloglik += llmbreg.sum()

    if torch.isnan(negloglik):
        print('unstable')

    return negloglik #, Vh


def negloglik_singlevar_gp_sp(llmb, lsigma2, theta, g):
    C0 = cormat(theta, theta, llmb)
    nug = torch.exp(lsigma2) / (1 + torch.exp(lsigma2))
    C = (1 - nug) * C0 + nug * torch.eye(theta.shape[0])  # (1 - nug) *

    W, V = torch.linalg.eigh(C)
    Vh = V / torch.sqrt(W.abs())
    fcenter = Vh.T @ g
    n = g.shape[0]

    sig2hat = (n * torch.mean(fcenter ** 2) + 10) / (n + 10)

    negloglik = 1/2 * torch.sum(torch.log(W.abs()))  # log-determinant
    negloglik += n/2 * torch.log(sig2hat)  # log of MLE of scale
    # negloglik += 1/2 * torch.sum(fcenter ** 2 / sig2hat)  # quadratic term

    # llmb, lsigma2 regularization
    llmbreg = 5 * (llmb + 1) ** 2
    llmbreg[-1] = 15 * llmb[-1] ** 2
    negloglik += llmbreg.sum() + 5 * (lsigma2 + 10)**2
    # negloglik += 1/2 * (llmb**2).sum() + 1/10 * (lsigma2 + 10)**2

    return negloglik