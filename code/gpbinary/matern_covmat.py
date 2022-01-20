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
    R = torch.ones((x1.shape[0], x2.shape[0])) * torch.exp(lmb[d-1])

    for j in range(d-1):
        S = torch.abs(x1[:, j].reshape(-1, 1) - x2[:, j]) / torch.exp(lmb[j])
        R *= (1 + S)
        V -= S

    R *= torch.exp(V)
    return R
