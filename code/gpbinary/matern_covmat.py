import torch


def covmat(x1, x2, lmb):
    '''

    :param x1:
    :param x2:
    :param lmb:
    :return:
    '''

    # assumes tensors are supplied
    d = lmb.shape[0]

    V = torch.zeros((x1.shape[0], x2.shape[0]))
    R = torch.ones((x1.shape[0], x2.shape[0])) * torch.exp(lmb[-1])

    for j in range(d-1):
        S = torch.abs(x1[:, j].reshape(-1, 1) - x2[:, j])
        R *= (1 + S)
        V -= S

    R *= torch.exp(V)

    return R
