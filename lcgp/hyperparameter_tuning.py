import torch


def parameter_clamping(t, trng, c:float=1.23):
    """
    Returns clamped hyperparameter between a given range.
    For each dimension, the parameter clamping follows
    $$  0.5 * (lb_i + ub_i) + 0.5 * (ub_i - lb_i) * tanh(c_i * (t_i - 0.5 * (ub_i - lb_i))). $$
    Coefficients c should be supplied.
    Examples:
    llmb (GP lengthscale): in [-2.5, 2.5].
    lsigma2 (noise variance): in [-12, -1].
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
