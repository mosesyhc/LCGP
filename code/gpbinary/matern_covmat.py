import torch


def covmat(x1, x2, lmb, diag_only=False):
    '''
    :param diag_only:
    :param x1:
    :param x2:
    :param lmb:
    :return:
    '''

    # assumes tensors are supplied
    assert x1.dim() == 2, 'input x1 should be 2-dimensional, (n_param, dim_param)'
    assert x2.dim() == 2, 'input x2 should be 2-dimensional, (n_param, dim_param)'
    d = lmb.shape[0]

    if diag_only:
        assert torch.isclose(x1, x2).all(), 'diag_only should only be called when x1 and x2 are identical.'
        c = torch.exp(lmb[d-1]) * torch.ones(x1.shape[0])
        return c

    else:
        V = torch.zeros((x1.shape[0], x2.shape[0]))
        C = torch.ones((x1.shape[0], x2.shape[0])) * torch.exp(lmb[d-1])

        for j in range(d-1):
            S = torch.abs(x1[:, j].reshape(-1, 1) - x2[:, j]) / torch.exp(lmb[j])
            C *= (1 + S)
            V -= S

        C *= torch.exp(V)
    return C


def cov_sp(theta, thetai, lsigma2, lmb):  # assuming x1 = x2 = theta
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
    C_full_diag = covmat(theta, theta, lmb=lmb, diag_only=True)

    W_i, U_i = torch.linalg.eigh(C_i)
    W_iinv = 1 / W_i
    C_iinv = U_i @ torch.diag(W_iinv) @ U_i.T

    C_r = c_full_i @ C_iinv @ c_full_i.T

    diag = C_full_diag - torch.diag(C_r) + torch.exp(lsigma2)
    Delta_inv_diag = 1 / diag


    R = C_i + (c_full_i.T * Delta_inv_diag) @ c_full_i  # p x p

    W_R, U_R = torch.linalg.eigh(R)
    Q_half = (Delta_inv_diag * c_full_i.T).T @ (U_R * torch.sqrt(1 / W_R)) @ U_R.T
    # Rinv_half = U_R @ torch.diag(torch.sqrt(1 / W_R)) @ U_R.T  # = Q_half = Lmb_inv @ c_full_i @ R_invhalf

    # C_sp_inv = torch.diag(Delta_inv_diag) - Q_half @ Q_half.T
    # C_sp_inv = Lmb_inv - Lmb_inv @ c_full_i @ Rinv @ c_full_i.T @ Lmb_inv  # improve to p x p matrices

    logdet_C_sp = torch.log(W_R).sum() - torch.log(W_i).sum() + torch.log(diag).sum()

    return Delta_inv_diag, Q_half, logdet_C_sp
