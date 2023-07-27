import numpy as np
import scipy.stats as sps


def rmse(y, ypredmean):
    return np.mean((y - ypredmean) ** 2)


def normalized_rmse(y, ypredmean):
    rng = (np.max(y, axis=1) - np.min(y, axis=1)).reshape(y.shape[0], 1)
    return np.mean(((y - ypredmean) / rng)**2)


def dss(y, ypredmean, ypredcov, use_diag):
    def __dss_single(f, mu, Sigma):  # Dawid-Sebastani score (1999)
        r = f - mu
        W, U = np.linalg.eigh(Sigma)
        r_Sinvh = r @ U * 1 / np.sqrt(W)

        _, logabsdet = np.linalg.slogdet(Sigma)

        score_single = logabsdet + (r_Sinvh ** 2).sum()
        return score_single

    def __dss_single_diag(f, mu, diagSigma):
        r = f - mu
        score_single = np.log(diagSigma).sum() + (r * r / diagSigma).sum()
        return score_single

    p, n = y.shape
    if use_diag:
        score_single = __dss_single_diag
    else:
        score_single = __dss_single

    score = 0
    for i in range(n):
        score += score_single(y[:, i], ypredmean[:, i], ypredcov[:, :, i])
    score /= n

    return score


def intervalstats(y, ypredmean, ypredvar):
    ylower = ypredmean + np.sqrt(ypredvar) * sps.norm.ppf(0.025)
    yupper = ypredmean + np.sqrt(ypredvar) * sps.norm.ppf(0.975)

    coverage = np.mean(np.logical_and(y <= yupper, y >= ylower))
    length = np.mean(yupper - ylower)
    return coverage, length
