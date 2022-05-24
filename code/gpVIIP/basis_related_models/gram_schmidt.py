import torch


def gram_schmidt_schwart(A):
    m, n = A.shape

    Q = torch.tensor(A)
    R = torch.zeros(n, n)

    for k in range(n):
        for i in range(k):
            R[i, k] = Q[:, i].T @ Q[:, k]
            Q[:, k] = Q[:, k] - R[i, k] * Q[:, i]
        R[k, k] = torch.linalg.norm(Q[:, k])
        Q[:, k] /= R[k, k]

    return -Q

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
    A = torch.randn((198, 50))
    Q = gram_schmidt(A)

    # print(torch.mean((Q.T @ Q - torch.eye(Q.shape[1]))**2))


def test_gs_schwart():
    A = torch.randn((198, 50))
    Q = gram_schmidt(A)

    # print(torch.mean((Q.T @ Q - torch.eye(Q.shape[1]))**2))


if __name__ == '__main__':
    import timeit
    print(timeit.Timer(test_gs).timeit(number=1000))
    print(timeit.Timer(test_gs_schwart).timeit(number=1000))
