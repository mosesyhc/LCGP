import torch


def gram_schmidt(vv):
    '''
    Returns an orthonormal matrix that spans the same range as the input.

    Modified from legendongary/pytorch-gram-schmidt.
    Modification includes debug for rectangular matrix.
    Copyright (c) 2019 legendongary
    '''
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[:, k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu

def test_gs():
    A = torch.randn((100, 10))
    Q2 = gram_schmidt(A)

    print(torch.mean((Q2.T @ Q2 - torch.eye(Q2.shape[1]))**2))


if __name__ == '__main__':
    test_gs()
