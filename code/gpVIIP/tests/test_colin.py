import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp
from surmise.emulation import emulator
from optim_elbo import optim_elbo, optim_elbo_lbfgs


res_struct = dict.fromkeys(['method', 'rep',
                            'n', 'p', 'seed',
                            'timeconstruct', 'Phi_mse',
                            'trainrmse', 'testrmse',
                            'optim_elbo_iter', 'optim_elbo_flag', 'optim_elbo_lr',
                            'precision'])


def build_surmise(ftr, thetatr, Phi):
    from surmise.emulation import emulator
    emu = emulator(x=np.arange(ftr.shape[0]),
                   theta=thetatr.numpy(),
                   f=ftr.numpy(), method='PCGPwM',
                   args={'warnings': True,
                         'standardpcinfo': {'U': Phi.numpy()}})

    return emu


def rmse_w_surmise(emu, fte, thetate):
    emupred = emu.predictmean(x=np.arange(ftr.shape[0]), theta=thetate.numpy())
    emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    return np.sqrt(emumse)


def test_single(method, n, seed, ftr, thetatr, fte, thetate, Phi,
                rep=None, ip_frac=None, precision_double=True,
                output_csv=False):
    res = res_struct.copy()
    if precision_double:
        torch.set_default_dtype(torch.float64)
        precision = 'float64'
    else:
        torch.set_default_dtype(torch.float32)
        precision = 'float32'

    p = n
    niter = None
    flag = None
    time_tr1 = None
    lr = 5e-4
    rmsetr = None
    rmsete = None

    Phi_mse = ((ftr - Phi @ Phi.T @ ftr) ** 2).mean()
    print('reconstruction mse: {:.3E}'.format(Phi_mse.numpy()))
    time_tr0 = time.time()
    # train model
    if method == 'surmise':
        emu = build_surmise(ftr, thetatr, Phi) #

        rmsetr = rmse_w_surmise(emu=emu, thetate=thetatr, fte=ftr)
        rmsete = rmse_w_surmise(emu=emu, thetate=thetate, fte=fte)

        time_tr1 = time.time()
        del emu
    elif method == 'MVGP':
        model = MVN_elbo_autolatent(Phi=Phi, F=ftr, theta=thetatr)
        model, niter, flag = optim_elbo_lbfgs(model,
                                              maxiter=100, lr=lr)

        time_tr1 = time.time()
        model.compute_MV()
        rmsetr = model.test_rmse(thetatr, ftr)
        rmsete = model.test_rmse(thetate, fte)

        del model
    elif method == 'MVIP':
        p = int(n * ip_frac)

        model = MVN_elbo_autolatent_sp(Phi=Phi, F=ftr, theta=thetatr, p=p) #, thetai=thetai)
        model, niter, flag = optim_elbo_lbfgs(model,
                                              maxiter=100,
                                              lr=lr)

        time_tr1 = time.time()

        model.compute_MV()
        rmsetr = model.test_rmse(thetatr, ftr)
        rmsete = model.test_rmse(thetate, fte)

        del model

    res['method'] = method
    res['rep'] = rep
    res['n'] = n
    res['p'] = p
    res['seed'] = seed

    res['Phi_mse'] = Phi_mse.numpy()
    res['timeconstruct'] = time_tr1 - time_tr0
    res['optim_elbo_iter'] = niter
    res['optim_elbo_lr'] = lr
    res['optim_elbo_flag'] = flag

    res['trainrmse'] = rmsetr.item()
    res['testrmse'] = rmsete.item()
    res['precision'] = precision

    print(rmsetr, rmsete)

    if output_csv:
        df = pd.DataFrame(res, index=[0])
        df.to_csv(res_dir + r'rep{:d}_n{:d}_p{:d}_{:s}_seed{:d}_{:s}.csv'.format(
            rep, n, p, method, int(seed), datetime.today().strftime('%Y%m%d%H%M%S'))
        )


if __name__ == '__main__':
    import pathlib
    res_dir = r'code/test_results/colin_comparison/comparison_20220519_kap10/'
    if not pathlib.Path(res_dir).exists():
        pathlib.Path(res_dir).mkdir(parents=True)
    data_dir = r'code/data/colin_data/'
    theta = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_input.csv')
    f = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_output.csv')
    theta = torch.tensor(theta.iloc[:, 1:].to_numpy())
    f = torch.tensor(f.iloc[:, 1:].to_numpy()).T

    # f = ((f.T - f.min(1).values) / (f.max(1).values - f.min(1).values)).T
    f = ((f.T - f.mean(1)) / f.std(1)).T

    # arbitrary x
    m, n_all = f.shape
    x = np.arange(m)

    ntr = 800
    indtr = torch.randperm(n_all)[:ntr]
    indte = np.setdiff1d(np.arange(n_all), indtr)
    ftr = f[:, indtr]
    thetatr = theta[indtr]
    fte = f[:, indte]
    thetate = theta[indte]


    method_list = ['MVIP', 'MVGP', 'surmise']
    n_list = [400, 800]
    ip_frac_list = [1/2, 1] # 1/4,

    nrep = 3
    # save_csv = True
    save_csv = False
    kap = 10
    for rep in np.arange(nrep):
        for n in n_list:
            seed = torch.randint(0, 10000, (1, ))
            torch.manual_seed(seed)
            tr_ind = torch.randperm(ntr)[:n]
            ftr_n = ftr[:, tr_ind]
            thetatr_n = thetatr[tr_ind]
            torch.seed()

            Phi, S, _ = torch.linalg.svd(ftr_n, full_matrices=False)
            Phi = Phi[:, :kap]
            S = S[:kap]

            for method in method_list:
                print('rep: {:d}, method: {:s}, n: {:d}'.format(rep, method, n))
                if method == 'MVIP':
                    for ip_frac in ip_frac_list:
                        print('ip_frac: {:.3f}'.format(ip_frac))
                        test_single(method=method, n=n, seed=seed,
                                    ftr=ftr_n, thetatr=thetatr_n,
                                    fte=fte, thetate=thetate,
                                    Phi=Phi,
                                    rep=rep, ip_frac=ip_frac,
                                    output_csv=save_csv)
                else:
                    test_single(method=method, n=n, seed=seed,
                                ftr=ftr_n, thetatr=thetatr_n,
                                fte=fte, thetate=thetate,
                                Phi=Phi,
                                rep=rep,
                                output_csv=save_csv)
