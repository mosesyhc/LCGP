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


if __name__ == '__main__':
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

    ntr = 200
    indtr = torch.randperm(n_all)[:ntr]
    indte = np.setdiff1d(np.arange(n_all), indtr)
    ftr = f[:, indtr]
    thetatr = theta[indtr]
    fte = f[:, indte]
    thetate = theta[indte]

    kap = 10
    Phi, _, _ = torch.linalg.svd(ftr, full_matrices=False)
    Phi = Phi[:, :kap]
    Phi_mse = ((ftr - Phi @ Phi.T @ ftr)**2).mean()

    model_full = MVN_elbo_autolatent(Phi=Phi, F=ftr, theta=thetatr)
    model_full, niter, flag = optim_elbo_lbfgs(model_full, lr=5e-3, ftol=model_full.n / 1e4)
    print(model_full.test_rmse(theta0=thetatr, f0=ftr))
    print(model_full.test_rmse(theta0=thetate, f0=fte))

    model_ip = MVN_elbo_autolatent_sp(Phi=Phi, F=ftr, theta=thetatr, p=ntr)
    model_ip, niter, flag = optim_elbo_lbfgs(model_ip, lr=5e-3, ftol=model_ip.n / 1e4)
    print(model_ip.test_rmse(theta0=thetatr, f0=ftr))
    print(model_ip.test_rmse(theta0=thetate, f0=fte))

    emu = emulator(f=ftr.numpy(),
                   theta=thetatr.numpy(),
                   x=x, method='PCGPwM')
    emupred = emu.predict(x=x, theta=thetate.numpy())
    emurmse = np.sqrt(((emupred.mean() - fte.numpy())**2).mean())
    print(emurmse)
