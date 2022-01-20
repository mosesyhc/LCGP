import torch
from matern_covmat import covmat
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
    R = covmat(theta, theta, lmb)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(W)  # check abs?

    Rinv_g = Vh @ Vh.T @ g
    Rnewold = covmat(thetanew, theta, lmb)
    Rnewnew = covmat(thetanew, thetanew, lmb)

    predmean = Rnewold @ Rinv_g
    predvar = Rnewnew - Rnewold @ Vh @ Vh.T @ Rnewold.T
    return predmean, predvar.diag()
