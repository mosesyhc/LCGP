import torch


C_LLMB = 0.4925
C_LSIGMA2 = 0.224


def clam_lmb(x, c=C_LLMB):
    return 2.5 * torch.tanh(c*(x))


def clam_lsigma2(x, c=C_LSIGMA2):
    return -13/2 + 11/2 * torch.tanh(c*(x + 13/2))


def parameter_clamping(t, trng, c):
    """
    Returns clamped hyperparameter between a given range.
    For each dimension, the parameter clamping follows
        $$  0.5 * (lb_i + ub_i) +
            0.5 * (ub_i - lb_i) * tanh(c_i * (t_i - 0.5 * (ub_i - lb_i))). $$
    Coefficients c should be supplied.
    Examples:
        llmb (GP lengthscale): in [-2.5, 2.5], optimal c ~= 0.4925, corresponding to an input in [0, 1].
        lsigma2 (noise variance): in [-12, -1], optimal c ~= 0.224

    :param t: d-dimensional, multiple parameters are stacked in columns.
    :param trng: either (lb, ub), or ((lb_1, ub_1), ..., (lb_d, ub_d))
    :param c: (c_1, c_2, ..., c_d)
    :return:
    """

    if trng.ndim < 2:
        l = trng[0]
        u = trng[1]
    else:
        l = trng[:, 0]
        u = trng[:, 1]

    return 0.5 * (u + l) + 0.5 * (u - l) * torch.tanh(c * (t - 0.5 * (u + l)))


if __name__ == '__main__':
    x = torch.column_stack((torch.arange(-14, 7, 0.5), torch.arange(-14, 7, 0.5)))
    trng = torch.row_stack((torch.tensor((-2.5, 2.5)), torch.tensor((-12, -1))))

    import matplotlib.pyplot as plt
    plt.style.use(['science', 'grid'])

    for i in torch.arange(0.05, 0.75, 0.005):
        plt.plot(x[:, 0], clam_lmb(x[:, 0], i), alpha=0.15, color='grey')
        plt.plot(x[:, 1], clam_lsigma2(x[:, 1], i), alpha=0.15, color='grey')
    plt.plot(x, x, linewidth=2.5, color='k', linestyle='dotted')
    # plt.scatter(x[:, 0], clam_lmb(x[:, 0], C_LLMB), color='r') #, label=r'c = {:.4f}'.format(C_LLMB))
    # plt.scatter(x[:, 1], clam_lsigma2(x[:, 1], C_LSIGMA2), color='r') #, label=r'c = {:.4f}'.format(C_LSIGMA2))
    plt.plot(x, parameter_clamping(x, trng, c=torch.tensor((C_LLMB, C_LSIGMA2))),
             label=(r'llmb', r'lsigma2'),  linewidth=2, alpha=0.75)
    plt.vlines(torch.tensor(((-12, -2.5), (-1, 2.5))), -14, 7, color=('g', 'b'), linestyle='dashed')
    plt.legend()
    plt.tight_layout()

