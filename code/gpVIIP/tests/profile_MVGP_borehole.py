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

from torch.profiler import profile, record_function, ProfilerActivity


res_struct = dict.fromkeys(['method', 'rep',
                            'n', 'p', 'seed',
                            'timeconstruct',
                            'trainrmse', 'testrmse', 'chi2', 'kap',
                            'optim_elbo_iter', 'optim_elbo_flag', 'optim_elbo_lr'])


def chi2metric(predmean, predstd, ftest):
    chi2 = (((predmean - ftest.numpy()) / predstd)**2).mean()
    return chi2


def test_single(method, n, seed, ftr, thetatr, fte, thetate,
                rep=None, ip_frac=None,
                output_csv=False):
    res = res_struct.copy()

    p = n
    niter = None
    flag = None
    lr = 1e-1
    time_tr1 = None
    rmsetr = None
    rmsete = None
    chi2 = None
    kap = None

    time_tr0 = time.time()
    # train model
    if method == 'MVGP':
        with record_function("model_init"):
            model = MVN_elbo_autolatent(F=ftr, theta=thetatr,
                                        clamping=True)
        kap = model.kap
        with record_function("model_optim"):
            model, niter, flag = optim_elbo_lbfgs(model,
                                                  maxiter=6, lr=lr)

        time_tr1 = time.time()
        with record_function("model_pred"):
            rmsetr = model.test_rmse(thetatr, ftr).item()
            rmsete = model.test_rmse(thetate, fte).item()

            predmean = model.predictmean(thetate).detach().numpy()
            predstd = model.predictvar(thetate).sqrt().detach().numpy()
            chi2 = chi2metric(predmean, predstd, fte)

        del model

    elif method == 'MVIP':
        p = int(n * ip_frac)

        with record_function("model_init"):
            model = MVN_elbo_autolatent_sp(F=ftr, theta=thetatr, p=p,
                                           clamping=True) #, thetai=thetai)
        kap = model.kap
        with record_function("model_optim"):
            model, niter, flag = optim_elbo_lbfgs(model,
                                                  maxiter=6,
                                                  lr=lr)

        time_tr1 = time.time()

        with record_function("model_pred"):
            rmsetr = model.test_rmse(thetatr, ftr).item()
            rmsete = model.test_rmse(thetate, fte).item()

            predmean = model.predictmean(thetate).detach().numpy()
            predstd = model.predictvar(thetate).sqrt().detach().numpy()
            chi2 = chi2metric(predmean, predstd, fte)

        del model

    res['method'] = method
    res['rep'] = rep
    res['n'] = n
    res['p'] = p
    res['seed'] = seed

    res['timeconstruct'] = time_tr1 - time_tr0
    res['optim_elbo_iter'] = niter
    res['optim_elbo_lr'] = lr
    res['optim_elbo_flag'] = flag

    res['trainrmse'] = rmsetr
    res['testrmse'] = rmsete
    res['chi2'] = chi2
    res['kap'] = kap

    if output_csv:
        df = pd.DataFrame(res, index=[0])
        df.to_csv(res_dir + r'rep{:d}_n{:d}_p{:d}_{:s}_seed{:d}_{:s}.csv'.format(
            rep, n, p, method, int(seed), datetime.today().strftime('%Y%m%d%H%M%S'))
        )


if __name__ == '__main__':
    import pathlib
    res_dir = r'code/test_results/borehole_comparisons/surmise_MVGP_MVIP/'
    dir = r'code/data/borehole_data/'
    f, x, thetatr = read_data(dir)

    m, ntr = f.shape
    fstd = f.std(1)
    ftr = np.zeros_like(f)
    for j in range(m):
        ftr[j] = f[j] + np.random.normal(0, 0.2 * fstd[j], ntr)

    fte, thetate = read_test_data(dir)

    ftr = torch.tensor(ftr)
    fte = torch.tensor(fte)
    thetatr = torch.tensor(thetatr)
    thetate = torch.tensor(thetate)

    method_list = ['MVGP', 'MVIP']  # 'surmise', 'MVIP',

    save_csv = False
    # save_csv = False
    n = 500
    rep = 0
    seed = torch.randint(0, 10000, (1, ))
    torch.manual_seed(seed.item())
    tr_ind = torch.randperm(ntr)[:n]
    ftr_n = ftr[:, tr_ind]
    thetatr_n = thetatr[tr_ind]
    torch.seed()

    for method in method_list:
        print('rep: {:d}, method: {:s}, n: {:d}'.format(rep, method, n))
        if method == 'MVIP':
            with profile(activities=[ProfilerActivity.CPU]) as profIP:
                test_single(method=method, n=n, seed=seed,
                            ftr=ftr_n, thetatr=thetatr_n,
                            fte=fte, thetate=thetate,
                            rep=rep, ip_frac=0.25,
                            output_csv=save_csv)
            profIP.export_chrome_trace('MVIP4.json')
        else:
            with profile(activities=[ProfilerActivity.CPU]) as profGP:
                test_single(method=method, n=n, seed=seed,
                            ftr=ftr_n, thetatr=thetatr_n,
                            fte=fte, thetate=thetate,
                            rep=rep,
                            output_csv=save_csv)
            profGP.export_chrome_trace('MVGP.json')

