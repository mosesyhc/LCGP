import torch
from matern_covmat import covmat, cov_sp
torch.set_default_dtype(torch.float64)


def pred_gp(lmb, theta, thetanew, g):
    '''
    Test in test_gp.py.

    :param lmb: hyperparameter for the covariance matrix
    :param theta: set of training parameters (size n x d)
    :param thetanew: set of testing parameters (size n0 x d)
    :param g: reduced rank latent variables (size n x 1)
    :return:
    '''

    # covariance matrix R for the training thetas
    C = covmat(theta, theta, lmb)

    W, V = torch.linalg.eigh(C)
    Vh = V / torch.sqrt(W)

    Cinv_g = Vh @ Vh.T @ g
    Cnewold = covmat(thetanew, theta, lmb)
    Cnewnew = covmat(thetanew, thetanew, lmb)

    predmean = Cnewold @ Cinv_g
    predvar = Cnewnew - Cnewold @ Vh @ Vh.T @ Cnewold.T
    return predmean, predvar.diag()



def pred_gp_sp(lmb, theta, thetanew, thetai, g):
    '''
    Test in test_gp.py.

    :param lmb: hyperparameter for the covariance matrix
    :param theta: set of training parameters (size n x d)
    :param thetanew: set of testing parameters (size n0 x d)
    :param g: reduced rank latent variables (size n x 1)
    :return:
    '''

    C, C_inv, _ = cov_sp(theta=theta, thetai=thetai, lmb=lmb)
    # covariance matrix R for the training thetas
    # R = covmat(theta, theta, lmb)

    # W, V = torch.linalg.eigh(R)
    # Vh = V / torch.sqrt(W)

    Cinv_g = C_inv @ g
    Cnewold = covmat(thetanew, theta, lmb)
    Cnewnew = covmat(thetanew, thetanew, lmb)

    predmean = Cnewold @ Cinv_g
    predvar = Cnewnew - Cnewold @ C_inv @ Cnewold.T
    return predmean, predvar.diag()
