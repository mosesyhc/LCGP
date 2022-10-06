import torch


def covmat(x1, x2, llmb, lnug, ltau2, diag_only:bool=False):
    # assumes tensors are supplied
    assert x1.dim() == 2, 'input x1 should be 2-dimensional, (n_param, dim_param)'
    assert x2.dim() == 2, 'input x2 should be 2-dimensional, (n_param, dim_param)'
    d = x1.shape[1]

    if diag_only:
        assert torch.isclose(x1, x2).all(), 'diag_only should only be called when x1 and x2 are identical.'
        c = torch.ones(x1.shape[0])  # / (1 + torch.exp(llmb[-1]))
        return ltau2.exp() * c

    else:
        V = torch.zeros((x1.shape[0], x2.shape[0]))
        C0 = torch.ones((x1.shape[0], x2.shape[0])) / (1 + torch.exp(llmb[-1]))

        x1scal = x1 / torch.exp(llmb[:-1])
        x2scal = x2 / torch.exp(llmb[:-1])
        for j in range(d):
            S = torch.abs(x1scal[:, j].reshape(-1, 1) - x2scal[:, j])  # outer diff
            C0 *= (1 + S)
            V -= S

        C0 *= torch.exp(V)
        C0 += torch.exp(llmb[-1]) / (1 + torch.exp(llmb[-1]))

        nug = lnug.exp() / (1 + lnug.exp())
        if torch.equal(x1, x2):
            C = (1 - nug) * C0 + nug * torch.eye(x1.shape[0])
        else:
            C = (1 - nug) * C0
        return ltau2.exp() * C


def cov_sp(theta, thetai, llmb, lnug, ltau2):  # assuming x1 = x2 = theta
    '''
    Returns the Nystr{\"o}m approximation of a covariance matrix,
    its inverse, and the log of its determinant.

    :param lnug:
    :param theta:
    :param thetai:
    :param llmb:
    :return:
    '''

    c_full_i = covmat(theta, thetai, llmb=llmb)
    C_i = covmat(thetai, thetai, llmb=llmb)
    # C_full_diag = cormat(theta, theta, llmb=llmb, diag_only=True)

    Wi, Ui = torch.linalg.eigh(C_i)
    Ciinvh = Ui / Wi.sqrt()
    # C_iinv = Uih @ Uih.T   # Change

    Crh = c_full_i @ Ciinvh
    # C_r = c_full_i @ C_iinv @ c_full_i.T

    nug = lnug.exp() / (1 + lnug.exp())
    diag = (1 - nug) * (1 - (Crh ** 2).sum(1)) + nug  #
    Delta_inv_diag = 1 / diag

    R = C_i + ((1 - nug) * c_full_i.T * Delta_inv_diag) @ c_full_i  # + lnugR.exp() * torch.ones(thetai.shape[0])

    WR, UR = torch.linalg.eigh(R)

    Rinvh = UR / WR.abs().sqrt()

    Q = (1 - nug).sqrt() * (Delta_inv_diag * c_full_i.T).T
    Q_Rinvh = Q @ Rinvh
    # Rinv_half = U_R @ torch.diag(torch.sqrt(1 / W_R)) @ U_R.T
    # Q_Rinvh = Delta_inv @ c_full_i @ R_invhalf

    # C_sp_inv = torch.diag(Delta_inv_diag) - Q_Rinvh @ Q_Rinvh.T
    # C_sp_inv = Delta_inv - Delta_inv @ c_full_i @ Rinv @ c_full_i.T @ Delta_inv  # improve to p x p matrices

    logdet_C_sp = torch.log(WR.abs()).sum() - torch.log(Wi).sum() + torch.log(diag).sum()
    return Delta_inv_diag, Q, Rinvh, Q_Rinvh, logdet_C_sp, c_full_i, C_i
