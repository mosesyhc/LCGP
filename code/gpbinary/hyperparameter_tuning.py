import torch


C_LLMB = 0.4925
C_LSIGMA2 = 0.224


def clam_llmb(x, c=C_LLMB):
    return 2.5 * torch.tanh(c*(x))


def clam_lsigma2(x, c=C_LSIGMA2):
    return -13/2 + 11/2 * torch.tanh(c*(x + 13/2))


def parameter_clamping(t, trng, c=1.23):
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

    return 0.5 * (u + l) + 0.5 * (u - l) * torch.tanh((2 * c / (u - l)) * (t - 0.5 * (u + l)))


if __name__ == '__main__':
    x = torch.column_stack((torch.arange(-5, 5, 0.5), torch.arange(-14, 1, 0.5*3/2)))
    trng = torch.row_stack((torch.tensor((-2.5, 2.5)), torch.tensor((-12, -1))))

    import matplotlib.pyplot as plt
    plt.style.use(['science', 'no-latex', 'grid'])
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    for i in torch.arange(0.01, 0.9, 0.005):
        ax[0].plot(x[:, 0], clam_llmb(x[:, 0], i), alpha=0.05, color='blue')
        ax[1].plot(x[:, 1], clam_lsigma2(x[:, 1], i), alpha=0.05, color='green')

    ax[0].plot(x[:, 0], x[:, 0], linewidth=2.5, color='k', linestyle='dotted')
    ax[0].plot(x[:, 0], parameter_clamping(x[:, 0], trng[0]),
             label=(r'$\log(\lambda_{k, l})$'), color='blue', linewidth=2, alpha=1)
    ax[0].vlines(torch.tensor((-2.5, 2.5)), -5, 5, linestyle='dashed')
    ax[0].set_xlabel(r'$\log(\lambda_{k, l})$')
    # ax[0].legend()

    ax[1].plot(x[:, 1], x[:, 1], linewidth=2.5, color='k', linestyle='dotted')
    ax[1].plot(x[:, 1], parameter_clamping(x[:, 1], trng[1]),
             label=(r'$\log(\sigma^2)$'), color='green', linewidth=2, alpha=1)
    ax[1].vlines(torch.tensor((-12, -1)), -14, 1, color='green', linestyle='dashed')
    ax[1].set_xlabel(r'$\log(\sigma^2)$')
    # ax[1].legend()

    plt.tight_layout()
    # plt.savefig(r'code\fig\clamping.png', dpi=150)
