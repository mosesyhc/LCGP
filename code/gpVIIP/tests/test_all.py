import time
from datetime import datetime
import pandas as pd
import torch
torch.set_num_threads(8)
from fayans_support import read_data, read_test_data
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp

from optim_elbo import optim_elbo, optim_elbo_lbfgs

res_struct = dict.fromkeys(['method', 'rep',
                            'n', 'p', 'seed',
                            'timeconstruct', 'Phi_mse',
                            'trainrmse', 'testrmse',
                            'optim_elbo_iter', 'optim_elbo_flag', 'optim_elbo_lr',
                            'precision'])


def test_single(method, n, seed, ftr, thetatr, fte, thetate,
                Phi, rep=None, ip_frac=None, precision_double=True,
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
    lr = None
    rmsetr = None
    rmsete = None

    time_tr0 = time.time()
    # train model
    if method == 'surmise':
        emu = build_surmise(ftr, thetatr, Phi) #

        rmsetr = rmse_w_surmise(emu=emu, thetate=thetatr, fte=ftr)
        rmsete = rmse_w_surmise(emu=emu, thetate=thetate, fte=fte)

        time_tr1 = time.time()
        del emu
    elif method == 'MVGP':
        lr = 5e-4
        model = MVN_elbo_autolatent(lLmb=None, initlLmb=True,
                                    lsigma2=None, initlsigma2=True,
                                    Phi=Phi, F=ftr, x=thetatr)
        model, niter, flag = optim_elbo_lbfgs(model, #ftr, thetatr, fte, thetate,
                                              maxiter=100, lr=lr)

        time_tr1 = time.time()
        model.compute_MV()
        rmsetr = model.test_rmse(thetatr, ftr)
        rmsete = model.test_rmse(thetate, fte)

        del model
    elif method == 'MVIP':
        lr = 5e-4
        p = int(n * ip_frac)

        model = MVN_elbo_autolatent_sp(lLmb=None, initlLmb=True,
                                       lsigma2=None, initlsigma2=True,
                                       p=p,
                                       Phi=Phi, F=ftr, theta=thetatr) #, thetai=thetai)
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

def build_surmise(ftr, thetatr, Phi=None):
    from surmise.emulation import emulator
    import numpy as np
    if Phi is None:
        emu = emulator(x=x0.numpy(), theta=thetatr.numpy(),
                       f=ftr.numpy(), method='PCGPwM',
                       args={'warnings': True})

    else:
        offset = ftr.mean(1)
        scale = np.ones(ftr.shape[0])
        S = np.ones(Phi.shape[1])
        fs = ((ftr.T - offset) / scale).T
        extravar = torch.mean((fs - Phi @ Phi.T @ fs) ** 2, 1) * (scale ** 2)
        standardpcinfo = {'offset': offset.numpy(),
                          'scale': scale,
                          'U': Phi.numpy(),
                          'S': S,
                          'fs': fs.T.numpy(),
                          'extravar': extravar.numpy()}

        emu = emulator(x=x0.numpy(), theta=thetatr.numpy(),
                       f=ftr.numpy(), method='PCGPwM',
                       args={'warnings': True,
                             'standardpcinfo': standardpcinfo})

    return emu


def rmse_w_surmise(emu, fte, thetate):
    import numpy as np
    emupred = emu.predictmean(x=x0.numpy(), theta=thetate.numpy())
    emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    return np.sqrt(emumse)


if __name__ == '__main__':
    import pathlib
    res_dir = r'code/test_results/comparison_20220515_jit_detach2/'
    if not pathlib.Path(res_dir).exists():
        pathlib.Path(res_dir).mkdir()

    data_dir = r'code/data/borehole_data/'
    f, x0, theta = read_data(data_dir)
    fte, thetate = read_test_data(data_dir)
    n_all = f.shape[1]

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)
    fte = torch.tensor(fte)
    thetate = torch.tensor(thetate)

    fmean = f.mean(1).unsqueeze(1)
    fstd = f.std(1).unsqueeze(1)

    f = (f - fmean) / fstd
    fte = (fte - fmean) / fstd

    ### list of methods
    method_list = [ 'MVIP', 'MVGP',  'surmise']
    n_list = [200, 400, 800] #,1600]  #, 1600] 200,
    ip_frac_list = [1/8, 1/4, 1/2, 1]  # 1/8, 1/4,

    save_csv = True
    # save_csv = False

    ### replication,
    nrep = 1
    kap = 5

    ### run test ###
    for rep in range(nrep):
        for n in n_list:
            seed = rep # torch.randint(0, 10000, (1,))
            ### train, test data
            torch.manual_seed(int(seed))
            tr_ind = torch.randperm(n_all)[:n]
            ftr = f[:, tr_ind]
            thetatr = theta[tr_ind]
            torch.seed()

            # construct basis
            Phi, S, _ = torch.linalg.svd(ftr, full_matrices=False)
            Phi = Phi[:, :kap]
            S = S[:kap]

            Phi_mse = ((ftr - Phi @ Phi.T @ ftr)**2).mean()

            for method in method_list:
                print('rep: {:d}, method: {:s}, n: {:d}'.format(rep, method, n))
                if method == 'MVIP':
                    for ip_frac in ip_frac_list:
                        print('ip_frac: {:.3f}'.format(ip_frac))
                        test_single(method=method, n=n, seed=seed,
                                    ftr=ftr, thetatr=thetatr,
                                    fte=fte, thetate=thetate,
                                    Phi=Phi,
                                    rep=rep, ip_frac=ip_frac,
                                    output_csv=save_csv)
                else:
                    test_single(method=method, n=n, seed=seed,
                                ftr=ftr, thetatr=thetatr,
                                fte=fte, thetate=thetate,
                                Phi=Phi,
                                rep=rep,
                                output_csv=save_csv)
