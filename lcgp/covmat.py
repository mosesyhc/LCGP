import torch


def Matern32(x1, x2, llmb, llmb0, lnug, diag_only: bool = False):
    """
    Returns the Matern 3/2 covariance matrix.

    :param x1: input 1 of size (number of inputs in x1, dimension of input)
    :param x2: input 2 of size (number of inputs in x2, dimension of input)
    :param llmb: log-lengthscale hyperparameter for each dimension
    :param llmb0: log-scale hyperparameter
    :param lnug: parameter to tune the nugget, nugget = exp(lnug) / (1 + exp(lnug))
    :param diag_only: returns diagonal of covariance matrix if True. Default to False.
    :return: covariance matrix of size (n1, n2)
    """
    # assumes tensors are supplied
    assert x1.dim() == 2, 'input x1 should be 2-dimensional, (n_param, dim_param)'
    assert x2.dim() == 2, 'input x2 should be 2-dimensional, (n_param, dim_param)'
    d = x1.shape[1]

    if diag_only:
        assert torch.isclose(x1, x2).all(), 'diag_only should only be called ' \
                                            'when x1 and x2 are identical.'
        c = torch.ones(x1.shape[0])  # / (1 + torch.exp(llmb[-1]))
        return llmb0.exp() * c

    else:
        V = torch.zeros((x1.shape[0], x2.shape[0]))
        C0 = torch.ones((x1.shape[0], x2.shape[0]))  # / (1 + torch.exp(llmb[-1]))

        x1scal = x1 / torch.exp(llmb)  # [:-1])
        x2scal = x2 / torch.exp(llmb)  # [:-1])
        for j in range(d):
            S = torch.abs(x1scal[:, j].reshape(-1, 1) - x2scal[:, j])  # outer diff
            C0 *= (1 + S)
            V -= S

        C0 *= torch.exp(V)
        # C0 += torch.exp(llmb[-1]) / (1 + torch.exp(llmb[-1]))

        nug = lnug.exp() / (1 + lnug.exp())
        if torch.equal(x1, x2):
            C = (1 - nug) * C0 + nug * torch.eye(x1.shape[0])
        else:
            C = (1 - nug) * C0
        return llmb0.exp() * C


def Matern32_sp(x, xi, llmb, llmb0, lnug):  # assuming x1 = x2 = theta
    '''
    Returns the Nystr{\"o}m approximation of a covariance matrix,
    its inverse, and the log of its determinant.

    :param x: input of size (number of inputs, dimension of input)
    :param xi: inducing inputs of size (number of inducing inputs, dimension of input)
    :param llmb: log-lengthscale hyperparameter for each dimension
    :param llmb0: log-scale hyperparameter
    :param lnug: parameter to tune the nugget, nugget = exp(lnug) / (1 + exp(lnug))
    :return: a covariance matrix of size (n, n)
    '''

    c_full_i = Matern32(x, xi, llmb=llmb, llmb0=llmb0, lnug=lnug)
    C_i = Matern32(xi, xi, llmb=llmb, llmb0=llmb0, lnug=lnug)
    C_full_diag = Matern32(x, x, llmb=llmb, llmb0=llmb0, lnug=lnug, diag_only=True)

    Wi, Ui = torch.linalg.eigh(C_i)
    Ciinvh = Ui / Wi.sqrt()

    Crh = c_full_i @ Ciinvh

    diag = C_full_diag - (Crh ** 2).sum(1)  #
    Delta_inv_diag = 1 / diag

    R = C_i + (c_full_i.T * Delta_inv_diag) @ c_full_i

    WR, UR = torch.linalg.eigh(R)
    Rinvh = UR / WR.abs().sqrt()

    Q = (Delta_inv_diag * c_full_i.T).T
    Q_Rinvh = Q @ Rinvh

    logdet_C_sp = torch.log(WR.abs()).sum() - torch.log(Wi).sum() \
                  + torch.log(diag).sum()
    return Delta_inv_diag, Q, Rinvh, Q_Rinvh, logdet_C_sp, c_full_i, C_i
