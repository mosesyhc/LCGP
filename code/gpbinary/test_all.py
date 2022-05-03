import time
from datetime import datetime
import pandas as pd
import torch
from fayans_support import read_data, read_test_data
from sklearn.cluster import KMeans
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp

from optim_elbo import optim_elbo

from matern_covmat import cov_sp, covmat

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
        emu = build_surmise(ftr, thetatr, Phi)

        time_tr1 = time.time()
        rmsetr = rmse_w_surmise(emu=emu, thetate=thetatr, fte=ftr)
        rmsete = rmse_w_surmise(emu=emu, thetate=thetate, fte=fte)

    elif method == 'MVGP':
        lr = 8e-3
        model = MVN_elbo_autolatent(lLmb=None, initlLmb=True,
                                    lsigma2=None, initlsigma2=True,
                                    Phi=Phi, F=ftr, theta=thetatr)
        model, niter, flag = optim_elbo(model, ftr, thetatr, fte, thetate,
                                        maxiter=100, lr=lr)

        time_tr1 = time.time()
        model.create_MV()
        rmsetr = model.test_rmse(thetatr, ftr)
        rmsete = model.test_rmse(thetate, fte)

    elif method == 'MVIP':
        lr = 5e-4
        p = int(n * ip_frac)

        kmeans_theta = KMeans(n_clusters=p, algorithm='full').fit(thetatr)
        thetai = torch.tensor(kmeans_theta.cluster_centers_)

        model = MVN_elbo_autolatent_sp(lLmb=None, initlLmb=True,
                                       lsigma2=None, initlsigma2=True,
                                       Phi=Phi, F=ftr, theta=thetatr, thetai=thetai)
        model, niter, flag = optim_elbo(model,
                                        ftr=ftr, thetatr=thetatr,
                                        fte=fte, thetate=thetate,
                                        maxiter=100,
                                        lr=lr)


        time_tr1 = time.time()
        # import matplotlib.pyplot as plt
        # plt.scatter(Mvar.detach().numpy(), S.detach().numpy())
        # plt.show()

        rmsetr = model.test_rmse(thetatr, ftr)
        rmsete = model.test_rmse(thetate, fte)

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

    res['trainrmse'] = rmsetr
    res['testrmse'] = rmsete
    res['precision'] = precision

    print(rmsetr, rmsete)

    if output_csv:
        df = pd.DataFrame.from_dict(res)
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
    emupred = emu.predict(x=x0.numpy(), theta=thetate.numpy())
    emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    return np.sqrt(emumse)


if __name__ == '__main__':
    res_dir = r'code/test_results/comparison_20220425/'
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
    method_list = ['MVIP', 'MVGP', 'surmise']
    n_list = [400, 800] #, 1600] 200,
    ip_frac_list = [1/4, 1/2, 1] # 1/8, 1/4,

    ### replication,
    nrep = 1
    kap = 5

    ### run test ###
    for rep in range(nrep):
        for n in n_list:
            seed = torch.randint(0, 10000, (1,))

            ### train, test data
            torch.manual_seed(int(seed))
            tr_ind = torch.randperm(n_all)[:n]
            ftr = f[:, tr_ind]
            thetatr = theta[tr_ind]
            torch.seed()

            # construct basis
            # Phi, Phi_mse = optim_Phi(ftr, kap)
            # Phi, Phi_mse = optim_Phi(ftr, kap)
            # print('S^2/n', S**2/n)
            Phi, S, _ = torch.linalg.svd(ftr, full_matrices=False)
            Phi = Phi[:, :kap]
            S = S[:kap]

            Phi_mse = ((Phi @ Phi.T @ ftr)**2).mean()

            for method in method_list:
                print('rep: {:d}, method: {:s}, n: {:d}'.format(rep, method, n))
                if method == 'MVIP':
                    for ip_frac in ip_frac_list:
                        print('ip_frac: {:.3f}'.format(ip_frac))
                        test_single(method=method, n=n, seed=seed,
                                    ftr=ftr, thetatr=thetatr,
                                    fte=fte, thetate=thetate,
                                    Phi=Phi,
                                    rep=rep, ip_frac=ip_frac)
                else:
                    test_single(method=method, n=n, seed=seed,
                                ftr=ftr, thetatr=thetatr,
                                fte=fte, thetate=thetate,
                                Phi=Phi,
                                rep=rep)