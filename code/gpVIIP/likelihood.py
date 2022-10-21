import torch
from matern_covmat import covmat, cov_sp


def negloglik_gp(llmb, lnug, ltau2, theta, g):
    C = covmat(theta, theta, llmb=llmb, lnug=lnug, ltau2=ltau2)

    W, V = torch.linalg.eigh(C)
    Vh = V / torch.sqrt(W.abs())
    fcenter = Vh.T @ g

    negloglik = 1/2 * torch.sum(torch.log(W.abs()))  # log-determinant
    negloglik += 1/2 * (fcenter**2).sum()

    # regularization of hyperparameter
    llmbreg = 1/2 * 10 * (llmb + 1) ** 2
    # llmbreg = 2 * ((5 + 1) * llmb + 5 / llmb.exp())  # invgamma(5, 5)

    llmbreg[-1] = 1/2 * 10 * llmb[-1] ** 2
    ltau2reg = 1/2 * 2 * ltau2**2

    negloglik += llmbreg.sum() + ltau2reg

    Cinvdiag = (Vh @ Vh.T).diag()

    return negloglik, Cinvdiag


def negloglik_gp_sp(llmb, lnug, ltau2, theta, thetai, g):
    Delta_inv_diag, _, _, QRinvh, logdet_C, _, _ = cov_sp(theta=theta, thetai=thetai, llmb=llmb, lnug=lnug, ltau2=ltau2)

    QRinvh_g = (QRinvh.T * g).sum(1)
    quad = (Delta_inv_diag * g ** 2).sum() - (QRinvh_g ** 2).sum()

    negloglik = 1/2 * logdet_C  # log-determinant
    negloglik += 1/2 * quad  # log of MLE of scale

    # regularization of hyperparameter
    llmbreg = 1/2 * 20 * (llmb + 1) ** 2
    llmbreg[-1] = 1/2 * 10 * llmb[-1] ** 2
    ltau2reg = 1/2 * 2 * ltau2**2

    negloglik += llmbreg.sum() + ltau2reg

    Cinvdiag = Delta_inv_diag - (QRinvh @ QRinvh.T).diag()

    return negloglik, Cinvdiag