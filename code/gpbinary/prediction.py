import torch
from matern_covmat import covmat, cov_sp
torch.set_default_dtype(torch.float64)


def pred_gp(lmb, theta, thetanew, lsigma2, g):
    '''
    Test in test_gp.py.

    :param lmb: hyperparameter for the covariance matrix
    :param theta: set of training parameters (size n x d)
    :param thetanew: set of testing parameters (size n0 x d)
    :param g: reduced rank latent variables (size n x 1)
    :return:
    '''

    # covariance matrix R for the training thetas
    C = covmat(theta, theta, lmb) + torch.diag(torch.exp(lsigma2) * torch.ones(theta.shape[0]))

    W, V = torch.linalg.eigh(C)
    Vh = V / torch.sqrt(W)

    Cinv_g = Vh @ Vh.T @ g
    Cnewold = covmat(thetanew, theta, lmb)
    Cnewnew = covmat(thetanew, thetanew, lmb)

    predmean = Cnewold @ Cinv_g
    predvar = Cnewnew - Cnewold @ Vh @ Vh.T @ Cnewold.T
    return predmean, predvar.diag()



def pred_gp_sp(lmb, theta, thetanew, thetai, lsigma2, g):
    '''
    Test in test_gp.py.

    :param lmb: hyperparameter for the covariance matrix
    :param theta: set of training parameters (size n x d)
    :param thetanew: set of testing parameters (size n0 x d)
    :param g: reduced rank latent variables (size n x 1)
    :return:
    '''


    Lmb_inv_diag, Q_half, _ = cov_sp(theta=theta, thetai=thetai, lsigma2=lsigma2, lmb=lmb)

    # C_inv = torch.diag(Lmb_inv_diag) - Q_half @ Q_half.T

    Cinv_g = Lmb_inv_diag * g - Q_half @ (Q_half.T * g).sum(1)
    Cnewold = covmat(thetanew, theta, lmb)
    Cnewnew = covmat(thetanew, thetanew, lmb)

    predmean = Cnewold @ Cinv_g
    predvar = Cnewnew - Cnewold @ (torch.diag(Lmb_inv_diag) - Q_half @ Q_half.T) @ Cnewold.T
    return predmean, predvar.diag()
