import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from fayans_support import read_data, read_test_data
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


def build_surmise(ftr, thetatr):
    from surmise.emulation import emulator
    emu = emulator(x=np.arange(ftr.shape[0]),
                   theta=thetatr.numpy(),
                   f=ftr.numpy(), method='PCGP',
                   args={'warnings': True})

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


def test_single(method, n, seed, ftr, thetatr, fte, thetate,
                fte0, noiseconst=None, rep=None, ip_frac=None,
                output_csv=False):
    res = res_struct.copy()

    p = n
    niter = None
    flag = None
    lr = 1e-1

    time_tr0 = time.time()
    # train model
    if method == 'surmise':
        emu = build_surmise(ftr, thetatr)  #
        kap = emu._info['pct'].shape[1]

        predmeantr = emu.predict().mean()
        emupred = emu.predict(x=np.arange(fte.shape[0]),
                              theta=thetate)
        predmean = emupred.mean()
        predcov = emupred.covx().transpose(2, 0, 1)
        predstd = np.sqrt(emupred.var())
        predaddvar = 0

        time_tr1 = time.time()

    elif method == 'MVGP':
        model = MVN_elbo_autolatent(F=ftr, theta=thetatr,
                                    clamping=True)
        kap = model.kap
        model, niter, flag = optim_elbo_lbfgs(model,
                                              maxiter=100, lr=lr)

        predmeantr = model.predictmean(thetatr).detach().numpy()
        predmean = model.predictmean(thetate).detach().numpy()
        predcov = model.predictcov(thetate).detach().numpy()

        n0 = thetate.shape[0]
        predvar = np.zeros((m, n0))
        for i in range(n0):
            predvar[:, i] = np.diag(predcov[:, :, i])
        predstd = np.sqrt(predvar)
        predaddvar = model.predictaddvar().detach().numpy()
        time_tr1 = time.time()

    elif method == 'MVIP':
        p = int(n * ip_frac)

        model = MVN_elbo_autolatent_sp(F=ftr, theta=thetatr, p=p,
                                       clamping=True) #, thetai=thetai)
        kap = model.kap
        model, niter, flag = optim_elbo_lbfgs(model,
                                              maxiter=100,
                                              lr=lr)

        predmeantr = model.predictmean(thetatr).detach().numpy()
        predmean = model.predictmean(thetate).detach().numpy()
        predcov = model.predictcov(thetate).detach().numpy()

        n0 = thetate.shape[0]
        predvar = np.zeros((m, n0))
        for i in range(n0):
            predvar[:, i] = np.diag(predcov[:, :, i])
        predstd = np.sqrt(predvar)
        predaddvar = model.predictaddvar().detach().numpy()
        time_tr1 = time.time()

    else:
        raise ValueError('Specify a valid method.')

    fte = fte.numpy()
    ftr = ftr.numpy()
    fte0 = fte0.numpy()
    dss = dss_individual(predmean=predmean, predcov=predcov, fte=fte)
    rmsetr = rmse(predmean=predmeantr, fte=ftr)
    rmsete = rmse(predmean=predmean, fte=fte)
    crps0 = crps(f=fte, mu=predmean, sigma=predstd)
    chi2 = chi2metric(predmean=predmean, predstd=predstd, fte=fte)

    ccover, cintwid, cintscore = interval_stats(mean=predmean, stdev=predstd, testf=fte0)
    pcover, pintwid, pintscore = interval_stats(mean=predmean, stdev=np.sqrt(((predstd**2).T + predaddvar).T), testf=fte)

    res['method'] = method
    res['rep'] = rep
    res['n'] = n
    res['p'] = p
    res['seed'] = seed
    res['noiseconst'] = noiseconst

    res['timeconstruct'] = time_tr1 - time_tr0
    res['optim_elbo_iter'] = niter
    res['optim_elbo_lr'] = lr
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
        df.to_csv(res_dir + r'rep{:d}_n{:d}_p{:d}_{:s}_seed{:d}_{:s}.csv'.format(
            rep, n, p, method, int(seed), datetime.today().strftime('%Y%m%d%H%M%S'))
        )


if __name__ == '__main__':
    from pathlib import Path
    res_dir = r'code/test_results/surmise_MVGP_MVIP/20221007/'
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    dir = r'code/data/borehole_data/'
    f, x, thetatr = read_data(dir)

    m, ntr = f.shape
    fstd = f.std(1)
    ftr = np.zeros_like(f)

    fte0, thetate = read_test_data(dir)
    fte = np.zeros_like(fte0)
    _, nte = fte.shape

    noiseconst = 3
    for j in range(m):
        ftr[j] = f[j] + np.random.normal(0, noiseconst * fstd[j], ntr)
        fte[j] = fte0[j] + np.random.normal(0, noiseconst * fstd[j], nte)

    ftr = torch.tensor(ftr)
    fte0 = torch.tensor(fte0)
    fte = torch.tensor(fte)
    thetatr = torch.tensor(thetatr)
    thetate = torch.tensor(thetate)

    method_list = ['surmise', 'MVIP', 'MVGP']
    n_list = [25, 50, 100, 250, 500] # 25, 50] #
    ip_frac_list = [1/4, 1/2, 1]

    nrep = 10
    save_csv = True
    for rep in np.arange(nrep):
        for n in n_list:
            seed = torch.randint(0, 10000, (1, ))
            torch.manual_seed(seed.item())
            tr_ind = torch.randperm(ntr)[:n]
            ftr_n = ftr[:, tr_ind]
            thetatr_n = thetatr[tr_ind]
            torch.seed()

            for method in method_list:
                print('rep: {:d}, method: {:s}, n: {:d}'.format(rep, method, n))
                if method == 'MVIP':
                    for ip_frac in ip_frac_list:
                        print('ip_frac: {:.3f}'.format(ip_frac))
                        test_single(method=method, n=n, seed=seed,
                                    ftr=ftr_n, thetatr=thetatr_n,
                                    fte=fte, fte0=fte0, thetate=thetate,
                                    noiseconst=noiseconst,
                                    rep=rep, ip_frac=ip_frac,
                                    output_csv=save_csv)
                else:
                    test_single(method=method, n=n, seed=seed,
                                ftr=ftr_n, thetatr=thetatr_n,
                                fte=fte, fte0=fte0, thetate=thetate,
                                noiseconst=noiseconst,
                                rep=rep,
                                output_csv=save_csv)
