import torch


def covmat(x1, x2, gammav, return_gradhyp=False):
    # assumes tensors are supplied
    d = gammav.shape[0]
    x1 = x1.reshape(1, d-1) / torch.exp(gammav[:-1]) if x1.ndim < 1.5 else x1 / torch.exp(gammav[:-1])
    x2 = x2.reshape(1, d-1) / torch.exp(gammav[:-1]) if x2.ndim < 1.5 else x2 / torch.exp(gammav[:-1])

    V = torch.zeros((x1.shape[0], x2.shape[0]))
    R = torch.ones((x1.shape[0], x2.shape[0])) * torch.exp(gammav[-1])
    S = torch.zeros((x1.shape[0], x2.shape[0]))

    if return_gradhyp:
        dR = torch.zeros((x1.shape[0], x2.shape[0], d))

    for j in range(d-1):
        S = torch.abs(x1[:, j].reshape(-1, 1) - x2[:, j])
        R *= (1 + S)
        V -= S

        if return_gradhyp:
            dR[:, :, j] = (S ** 2) / (1 + S)

    if return_gradhyp:
        dR *= R.unsqueeze(2)
        dR[:, :, -1] = R
    R *= torch.exp(V)

    if return_gradhyp:
        return R, dR
    else:
        return R
