import numpy as np
from scipy import optimize as spo

def negloglik(S_flatten, y):
    n, p = y.shape
    y = y.astype(int)

    states, counts = np.unique(y, return_counts=True, axis=0)
    offdiag = (-1) ** (1 - states)

    assert S_flatten.size == p*p
    S = S_flatten.reshape((p, p))

    nll = 0
    for i in np.arange(states.shape[0]):
        G = np.multiply(S, np.tile(offdiag[i, :], (p, 1)))
        svec = np.diag(S)
        np.fill_diagonal(G, (svec ** states[i, :]) * (1 - svec) ** (1 - states[i, :]))
        (sign, logdetG) = np.linalg.slogdet(G)
        nll += -counts[i] * sign * logdetG

    return nll


def negloglikgrad(S_flatten, y):
    n, p = y.shape
    y = y.astype(int)

    states, counts = np.unique(y, return_counts=True, axis=0)
    offdiag = (-1) ** (1 - states)

    assert S_flatten.size == p*p
    S = S_flatten.reshape((p, p))

    grad = np.zeros_like(S)
    for i in np.arange(states.shape[0]):
        G = np.multiply(S, np.tile(offdiag[i, :], (p, 1)))
        grad += -counts[i] * np.linalg.inv(G)

    return grad.flatten()


def negloglik2(S_flatten, y):
    # 9 secs to compute likelihood with 1000 x 198
    n, p = y.shape

    y = y.astype(int)
    offdiag = (-1) ** (1 - y)

    assert S_flatten.size == p*p
    S = S_flatten.reshape((p, p))

    nll = 0
    for i in np.arange(n):
        G = np.multiply(S, np.tile(offdiag[i, :], (p, 1)))
        svec = np.diag(S)
        np.fill_diagonal(G, (svec ** y[i, :]) * (1 - svec) ** (1 - y[i, :]))
        (sign, logdetG) = np.linalg.slogdet(G)
        nll += -sign * logdetG
    return nll


def check_compute_time():
    f = np.loadtxt(r'code/f.txt')

    y = np.isnan(f).T

    S = np.ones((y.shape[1], y.shape[1])) * 0.1
    np.fill_diagonal(S, 0.8)

    # time check
    import time
    start = time.time()
    for i in np.arange(10):
        negloglik(S.flatten(), y)
    end = time.time()

    fasttime = (end - start) / 10

    start2 = time.time()
    for i in np.arange(10):
        negloglik2(S.flatten(), y)
    end2 = time.time()
    slowtime = (end2 - start2) / 10

    print('Fast: {:.3f} seconds'.format(fasttime))
    print('Slow: {:.3f} seconds'.format(slowtime))


if __name__ == '__main__':
    y = np.random.choice((0, 1), size=(1000, 5), replace=True, p=(0.2, 0.8))

    n, p = y.shape
    diag = np.ones(p) * y.mean(0)
    S = np.ones((p, p)) * 0.5
    np.fill_diagonal(S, diag)

    lb = np.full_like(S, np.nan)
    np.fill_diagonal(lb, np.zeros(p))
    ub = np.full_like(S, np.nan)
    np.fill_diagonal(ub, np.ones(p))
    bounds = spo.Bounds(lb.flatten(), ub.flatten())
    opt = spo.minimize(negloglik, S.flatten(), y, method='L-BFGS-B', jac=negloglikgrad, bounds=bounds)
