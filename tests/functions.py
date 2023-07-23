import numpy as np

# Cox, Parker, and Singer (2001)
def cps2001(x):
    x = x.expand_dims(0) if x.ndim < 2 else x
    x1, x2, x3, x4 = [x[:, i] for i in range(x.shape[1])]

    y11 = (x1 / 2) * (np.sqrt(1 + (x2 + x3**2) * x4 / x1**2) - 1)
    y12 = (x1 + 3*x4) * np.exp(1 + np.sin(x3))
    y1 = y11 + y12

    y2 = (1 + np.sin(x1)/10)*y1.copy() - 2*x1 + x2**2 + x3**2 + 0.5

    e1 = np.random.normal(0, 5 * x.mean(1) ** 2, x.shape[0])
    e2 = np.random.normal(0, 5 * x.mean(1) ** 2, x.shape[0])

    y1 += e1
    y2 += e2

    y = np.column_stack((y1, y2))
    return y


def forrester2008(x, noisy=True, noises=(0.01, 0.1, 0.25)):
    x = np.expand_dims(x, 1) if x.ndim < 2 else x

    y1 = (6*x - 2)**2 * np.sin(12*x - 4)
    def forrester1d(y, x, a, b, c):
        return a*y + b*(x - 0.5) - c
    y2 = forrester1d(y1, x, 0.5, 5, -5)
    y3 = forrester1d(y1, x, -0.8, -5, 4)
    if noisy:
        e1 = np.random.normal(0, np.sqrt(noises[0]) * np.std(y1), x.shape)
        e2 = np.random.normal(0, np.sqrt(noises[1]) * np.std(y2), x.shape)
        e3 = np.random.normal(0, np.sqrt(noises[2]) * np.std(y3), x.shape)
        y1 += e1
        y2 += e2
        y3 += e3
    y = np.row_stack((y1.T, y2.T, y3.T))

    return y
