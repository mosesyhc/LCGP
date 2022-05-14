import torch
from hyperparameter_tuning import parameter_clamping


def covmat(x1, x2, llmb, diag_only=False):
    '''
    :param diag_only:
    :param x1:
    :param x2:
    :param llmb:
    :return:
    '''

    # assumes tensors are supplied
    assert x1.dim() == 2, 'input x1 should be 2-dimensional, (n_param, dim_param)'
    assert x2.dim() == 2, 'input x2 should be 2-dimensional, (n_param, dim_param)'
    d = llmb.shape[0]

    if diag_only:
        assert torch.isclose(x1, x2).all(), 'diag_only should only be called when x1 and x2 are identical.'
        c = torch.exp(llmb[d - 1]) * torch.ones(x1.shape[0])
        return c

    else:
        V = torch.zeros((x1.shape[0], x2.shape[0]))
        C = torch.ones((x1.shape[0], x2.shape[0])) * torch.exp(llmb[d - 1])

        for j in range(d-1):
            S = torch.abs(x1[:, j].reshape(-1, 1) - x2[:, j]) / torch.exp(llmb[j])
            C *= (1 + S)
            V -= S

        C *= torch.exp(V)
    return C


def cov_sp(theta, thetai, llmb):  # assuming x1 = x2 = theta
    '''
    Returns the Nystr{\"o}m approximation of a covariance matrix,
    its inverse, and the log of its determinant.

    :param theta:
    :param thetai:
    :param llmb:
    :return:
    '''

    c_full_i = covmat(theta, thetai, llmb=llmb)
    C_i = covmat(thetai, thetai, llmb=llmb)
    C_full_diag = covmat(theta, theta, llmb=llmb, diag_only=True)

    W_Ci, U_Ci = torch.linalg.eigh(C_i)
    W_Ciinv = 1 / W_Ci
    C_iinv = U_Ci @ torch.diag(W_Ciinv) @ U_Ci.T

    C_r = c_full_i @ C_iinv @ c_full_i.T

    diag = C_full_diag - torch.diag(C_r) + torch.exp(lsigma2)
    Delta_inv_diag = 1 / diag


    R = C_i + (c_full_i.T * Delta_inv_diag) @ c_full_i  # p x p

    W_R, U_R = torch.linalg.eigh(R)
    Q_half = (Delta_inv_diag * c_full_i.T).T @ (U_R * torch.sqrt(1 / W_R.abs())) @ U_R.T
    # Rinv_half = U_R @ torch.diag(torch.sqrt(1 / W_R)) @ U_R.T  # = Q_half = Lmb_inv @ c_full_i @ R_invhalf

    # C_sp_inv = torch.diag(Delta_inv_diag) - Q_half @ Q_half.T
    # C_sp_inv = Lmb_inv - Lmb_inv @ c_full_i @ Rinv @ c_full_i.T @ Lmb_inv  # improve to p x p matrices

    logdet_C_sp = torch.log(W_R).sum() - torch.log(W_i).sum() + torch.log(diag).sum()

    return Delta_inv_diag, Q_half, logdet_C_sp
