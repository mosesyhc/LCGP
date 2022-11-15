import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp
from optim_elbo import optim_elbo_lbfgs
torch.set_default_dtype(torch.float64)

res_struct = dict.fromkeys(['method', 'rep',
                            'n', 'p', 'seed', 'noiseconst',
                            'timeconstruct',
                            'trainrmse', 'testrmse', 'chi2',
                            'kap', 'crps', 'dss',
                            'ccover', 'cintwid', 'cintscore',
                            'pcover', 'pintwid', 'pintscore',
                            'optim_elbo_iter', 'optim_elbo_flag', 'optim_elbo_lr'])


def build_surmise(ftr, xtr):
    from surmise.emulation import emulator
    emu = emulator(x=np.arange(ftr.shape[0]),
                   theta=xtr.numpy(),
                   f=ftr.numpy(), method='PCGPwM',
                   args={'warnings': True,
                         'nmaxhyptrain': 2000})
    return emu

def crps(f, mu, sigma):
    from scipy.stats import norm
    z = (f - mu) / sigma

    pz = norm.pdf(z)
    cz = norm.cdf(z)

    s = -sigma * (1/np.sqrt(np.pi) - 2 * pz - z * (2 * cz - 1))
    return s.mean()

def interval_stats(mean, stdev, testf, pr=0.95):
    from scipy.stats import norm
    alph = 1 - pr

    f = testf

    ci = np.zeros((2, *mean.shape))
    ci[0] = mean + norm.ppf(alph / 2) * stdev
    ci[1] = mean + norm.ppf(1 - alph / 2) * stdev

    under = np.less(f, ci[0])
    over = np.greater(f, ci[1])

    coverage = (1 - np.logical_or(under, over)).mean()
    avgintwidth = (ci[1] - ci[0]).mean()
    intScore = np.mean((ci[1] - ci[0]) +
                          2 / alph * (ci[0] - f) * under +
                          2 / alph * (f - ci[1]) * over)
    return coverage, avgintwidth, intScore


def chi2metric(predmean, predstd, fte):
    chi2 = (((predmean - fte) / predstd)**2).mean()
    return chi2


def rmse(predmean, fte):
    mse = np.mean(((predmean - fte) ** 2))
    return np.sqrt(mse)


def dss_individual(predmean, predcov, fte):
    def __dss_single_diag(f, mu, diagS):
        r = f - mu
        score_single = np.log(diagS).sum() + (r * r / diagS).sum()
        return score_single

    n0 = fte.shape[1]

    score = 0
    for i in range(n0):
        score += __dss_single_diag(f=fte[:, i], mu=predmean[:, i], diagS=np.diag(predcov[:, :, i]))
    score /= n0

    return score


def test_single(method, fname, n, seed, ftr, xtr, fte, xte,
                fte0, noiseconst=None, rep=0, ip_frac=None,
                output_csv=False, dir=None, return_quant=False):
    res = res_struct.copy()

    p = n
    niter = None
    flag = None

    time_tr0 = time.time()
    # train model
    if method == 'surmise':
        model = build_surmise(ftr, xtr)  #
        kap = model._info['pcto'].shape[1]
        pct = model._info['pcto']

        predmeantr = model.predict().mean()
        emupred = model.predict(x=np.arange(fte.shape[0]),
                                theta=xte)
        predmean = emupred.mean()
        predcov = emupred.covx().transpose(2, 0, 1)
        predstd = np.sqrt(emupred.var())
        predaddvar = model._info['standardpcinfo']['extravar']  # This corresponds to surmise PCGPwM develop version (after standardpcinfo)

        time_tr1 = time.time()

    elif method == 'MVGP':
        model = MVN_elbo_autolatent(F=ftr, x=xtr,
                                    clamping=True)
        pct = model.Phi
        kap = model.kap
        niter, flag = model.fit()
        predmeantr = model.predictmean(xtr).detach().numpy()
        predmean = model.predictmean(xte).detach().numpy()
        predcov = model.predictcov(xte).detach().numpy()

        m, n0 = fte.shape
        predvar = np.zeros((m, n0))
        for i in range(n0):
            predvar[:, i] = np.diag(predcov[:, :, i])
        predstd = np.sqrt(predvar)

        predaddvar = model.predictaddvar().detach().numpy()
        time_tr1 = time.time()
    #
    # elif method == 'MVIP':
    #     p = int(n * ip_frac)
    #
    #     model = MVN_elbo_autolatent_sp(F=ftr, theta=xtr, p=p,
    #                                    clamping=True) #, thetai=thetai)
    #     pct = model.Phi
    #     kap = model.kap
    #     model, niter, flag = optim_elbo_lbfgs(model,
    #                                           maxiter=100,
    #                                           lr=lr)
    #
    #
    #     predmeantr = model.predictmean(xtr).detach().numpy()
    #     predmean = model.predictmean(xte).detach().numpy()
    #     predcov = model.predictcov(xte).detach().numpy()
    #
    #     m, n0 = fte.shape
    #     predvar = np.zeros((m, n0))
    #     for i in range(n0):
    #         predvar[:, i] = np.diag(predcov[:, :, i])
    #     predstd = np.sqrt(predvar)
    #     predaddvar = model.predictaddvar().detach().numpy()
    #     time_tr1 = time.time()

    else:
        raise ValueError('Specify a valid method.')

    fte = fte.numpy()
    ftr = ftr.numpy()
    fte0 = fte0.numpy()
    dss = dss_individual(predmean=predmean, predcov=predcov, fte=fte0)
    rmsetr = rmse(predmean=predmeantr, fte=ftr)
    rmsete = rmse(predmean=predmean, fte=fte0)
    crps0 = crps(f=fte0, mu=predmean, sigma=predstd)
    chi2 = chi2metric(predmean=predmean, predstd=predstd, fte=fte0)

    ccover, cintwid, cintscore = interval_stats(mean=predmean, stdev=predstd, testf=fte0)
    pcover, pintwid, pintscore = interval_stats(mean=predmean, stdev=np.sqrt(((predstd**2).T + predaddvar).T), testf=fte)

    res['method'] = method
    res['fname'] = fname
    res['rep'] = rep
    res['n'] = n
    res['p'] = p
    res['seed'] = seed
    res['noiseconst'] = noiseconst

    res['timeconstruct'] = time_tr1 - time_tr0
    res['optim_elbo_iter'] = niter
    res['optim_elbo_lr'] = 'default'
    res['optim_elbo_flag'] = flag

    res['trainrmse'] = rmsetr
    res['testrmse'] = rmsete
    res['chi2'] = chi2
    res['kap'] = kap
    res['dss'] = dss
    res['crps'] = crps0

    res['ccover'] = ccover
    res['cintwid'] = cintwid
    res['cintscore'] = cintscore
    res['pcover'] = pcover
    res['pintwid'] = pintwid
    res['pintscore'] = pintscore

    if output_csv:
        df = pd.DataFrame(res, index=[0])
        df.to_csv(dir + r'{:s}_rep{:d}_n{:d}_p{:d}_{:s}_seed{:d}_{:s}.csv'.format(
            fname, rep, n, p, method, int(seed), datetime.today().strftime('%Y%m%d%H%M%S'))
        )

    if return_quant:
        return model, predmean, predstd, pct, n, xtr, ftr, xte, fte
    else:
        return