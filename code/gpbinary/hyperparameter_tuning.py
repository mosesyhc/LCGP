import torch


def parameter_clamping(theta, thetarng, rule='tanh'):
    '''
    Returns transformed parameters (theta) that falls into range (thetarng).

    :param theta: d-dimensional
    :param thetarng: either (lb, ub), or ((lb_1, ub_1), ..., (lb_d, ub_d))
    :param rule:
    :return:
    '''

