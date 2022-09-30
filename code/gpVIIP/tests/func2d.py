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


def forrester2008(x, noisy=True, noiseconst=1):
    x = np.expand_dims(x, 1) if x.ndim < 2 else x

    y1 = (6*x - 2)**2 * np.sin(12*x - 4)
    def forrester1d(y, x, a, b, c):
        return a*y + b*(x - 0.5) - c
    y2 = forrester1d(y1, x, 0.5, 5, -5)

    if noisy:
        e1 = np.random.normal(0, noiseconst * (x + 0.5) ** 2, x.shape)
        e2 = np.random.normal(0, noiseconst * (x + 0.5) ** 2, x.shape)

        y1 += e1
        y2 += e2
    y = np.row_stack((y1.T, y2.T))

    return y


def plot_forrester():
    import matplotlib.pyplot as plt
    plt.style.use(['science', 'grid'])

    x = np.linspace(0, 1, 500)
    f = forrester2008(x, noisy=False)

    # plt.figure(figsize=(8, 6))
    # plt.plot(x, f.T, linewidth=3)
    # plt.legend(labels=[r'$f_1(x) = (6x - 2)^2 \sin(12x - 4)$',
    #                    r'$f_2(x) = 0.5 f_1(x) + 5 (x - 0.5) + 5$'],
    #            fontsize='x-large')
    # plt.title('Forrester et al. (2008) function', fontsize=24)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.xlabel(r'$\boldsymbol x$', fontsize=20)
    # plt.ylabel(r'$\boldsymbol f(\boldsymbol x)$', fontsize=20, rotation=0)
    # plt.tight_layout()
    # plt.savefig('forrester2d.png', dpi=300)
    # plt.close()

    x0 = np.random.uniform(0, 1, 25)
    Phi, S, _ = np.linalg.svd(forrester2008(x0, noisy=True))
    Phi = Phi[:, :1]

    plt.figure(figsize=(8, 6))
    plt.plot(x, -(Phi.T @ f).T, linewidth=3)
    plt.title(r'Projected function', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'$\boldsymbol x$', fontsize=20)
    plt.ylabel(r'$g(\boldsymbol x)$', fontsize=20, rotation=0, labelpad=12)
    plt.tight_layout()
    plt.savefig('projected.png', dpi=300)
    plt.close()

# plot_forrester()
