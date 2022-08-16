import torch
from matern_covmat import cormat, cov_sp

torch.set_default_dtype(torch.float64)


def pred_gp(llmb, lsigma2, theta, thetanew, g):
    '''
    Test in test_gp.py.

    :param llmb: hyperparameter for the covariance matrix
    :param theta: set of training parameters (size n x d)
    :param thetanew: set of testing parameters (size n0 x d)
    :param g: reduced rank latent variables (size n x 1)
    :return:
    '''

    # covariance matrix R for the training thetas
    C0 = cormat(theta, theta, llmb)
    nug = torch.exp(lsigma2) / (1 + torch.exp(lsigma2))

    C = (1 - nug) * C0 + nug * torch.eye(theta.shape[0])

    W, V = torch.linalg.eigh(C)
    Vh = V / torch.sqrt(W)

    Cinv_g = Vh @ Vh.T @ g
    Cnewold = cormat(thetanew, theta, llmb)
    Cnewnew = cormat(thetanew, thetanew, llmb)

    predmean = Cnewold @ Cinv_g
    predvar = Cnewnew - Cnewold @ Vh @ Vh.T @ Cnewold.T
    return predmean, predvar.diag() + nug


def pred_gp_sp(llmb, theta, thetanew, thetai, g):
    '''
    Test in test_gp.py.

    :param llmb: hyperparameter for the covariance matrix
    :param theta: set of training parameters (size n x d)
    :param thetanew: set of testing parameters (size n0 x d)
    :param g: reduced rank latent variables (size n x 1)
    :return:
    '''

    Delta_inv_diag, Q_half, _, _, _ = cov_sp(theta=theta, thetai=thetai, llmb=llmb)

    # C_inv = torch.diag(Lmb_inv_diag) - Q_half @ Q_half.T

    Cinv_g = Delta_inv_diag * g - Q_half @ (Q_half.T * g).sum(1)
    Cnewold = cormat(thetanew, theta, llmb)
    Cnewnew = cormat(thetanew, thetanew, llmb)

    predmean = Cnewold @ Cinv_g
    predvar = Cnewnew - Cnewold @ (torch.diag(Delta_inv_diag) - Q_half @ Q_half.T) @ Cnewold.T
    return predmean, predvar.diag()
