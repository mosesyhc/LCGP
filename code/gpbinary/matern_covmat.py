import torch


def covmat(x1, x2, lmb):
    '''
    :param x1:
    :param x2:
    :param lmb:
    :return:
    '''

    # assumes tensors are supplied
    assert x1.dim() == 2, 'input x1 should be 2-dimensional, (n_param, dim_param)'
    assert x2.dim() == 2, 'input x2 should be 2-dimensional, (n_param, dim_param)'
    d = lmb.shape[0]

    V = torch.zeros((x1.shape[0], x2.shape[0]))
    C = torch.ones((x1.shape[0], x2.shape[0])) * torch.exp(lmb[d-1])

    for j in range(d-1):
        S = torch.abs(x1[:, j].reshape(-1, 1) - x2[:, j]) / torch.exp(lmb[j])
        C *= (1 + S)
        V -= S

    C *= torch.exp(V)
    return C


def cov_sp(theta, thetai, lmb):  # assuming x1 = x2 = theta
    '''
    Returns the Nystr{\"o}m approximation of a covariance matrix,
    its inverse, and the log of its determinant.

    :param theta:
    :param thetai:
    :param lmb:
    :return:
    '''

    # If thetai is a subset of theta,
    # caution on negative non-zero residuals that causes log(-ve) to be nan.

    c_full_i = covmat(theta, thetai, lmb=lmb)
    C_i = covmat(thetai, thetai, lmb=lmb)
    C_full = covmat(theta, theta, lmb=lmb)

    W_i, U_i = torch.linalg.eigh(C_i)
    W_iinv = 1 / W_i
    C_iinv = U_i @ torch.diag(W_iinv) @ U_i.T

    C_r = c_full_i @ C_iinv @ c_full_i.T

    diag = torch.diag(C_full - C_r)
    # diag = torch.clip(diag, 0.0, torch.inf) + 10e-8 # only matters when thetai is a subset of theta
    diag_inv = 1 / diag

    C_sp = C_r + torch.diag(diag)

    D_inv = torch.diag(diag_inv)
    R = (C_i + c_full_i.T @ D_inv @ c_full_i)

    W_R, U_R = torch.linalg.eigh(R)
    W_Rinv = 1 / W_R
    Rinv = U_R @ torch.diag(W_Rinv) @ U_R.T

    C_sp_inv = D_inv - D_inv @ c_full_i @ Rinv @ c_full_i.T @ D_inv

    # print(torch.log(W_R), torch.log(W_i), torch.log(diag))

    logdet_C_sp = torch.log(W_R).sum() - torch.log(W_i).sum() + torch.log(diag).sum()

    return C_sp, C_sp_inv, logdet_C_sp
