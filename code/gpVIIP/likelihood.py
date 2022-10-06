import torch
import torch.distributions.normal as Normal
from matern_covmat import covmat, cov_sp
norm = Normal.Normal(loc=0, scale=1)


def negloglik_gp(llmb, lnug, ltau2, theta, g):
    C = covmat(theta, theta, llmb=llmb, lnug=lnug, ltau2=ltau2)

    W, V = torch.linalg.eigh(C)
    Vh = V / torch.sqrt(W.abs())
    fcenter = Vh.T @ g

    negloglik = 1/2 * torch.sum(torch.log(W.abs()))  # log-determinant
    negloglik += 1/2 * (fcenter**2).sum()

    # regularization of hyperparameter
    llmbreg = 1/2 * 10 * (llmb + 1) ** 2
    llmbreg[-1] = 1/2 * 20 * llmb[-1] ** 2
    ltau2reg = 1/2 * 2 * ltau2**2

    negloglik += llmbreg.sum() + ltau2reg

    Cinvdiag = (Vh @ Vh.T).diag()

    return negloglik, Cinvdiag


def negloglik_gp_sp(llmb, lnug, theta, thetai, g):
    Delta_inv_diag, _, _, QRinvh, logdet_C, _, _ = cov_sp(theta, thetai, llmb, lnug)

    n = g.shape[0]
    QRinvh_g = (QRinvh.T * g).sum(1)
    quad = g @ (Delta_inv_diag * g) - (QRinvh_g ** 2).sum()
    tau2hat = (quad + 10) / (n + 10)
    # print(tau2hat)

    negloglik = 1/2 * logdet_C  # log-determinant
    negloglik += n/2 * torch.log(tau2hat)  # log of MLE of scale

    # llmb, lsigma2 regularization
    llmbreg = 15 * (llmb + 1) ** 2
    llmbreg[-1] = 25 * llmb[-1] ** 2
    negloglik += llmbreg.sum()

    if torch.isnan(negloglik):
        print('unstable')

    return negloglik #, Vh