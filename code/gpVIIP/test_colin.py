import numpy as np
import pandas as pd
import torch
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from mvn_elbo_autolatent_sp_model_jit import MVN_elbo_autolatent_sp
from surmise.emulation import emulator
from optim_elbo import optim_elbo, optim_elbo_lbfgs


if __name__ == '__main__':
    data_dir = r'code/data/colin_data/'
    theta = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_input.csv')
    f = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_output.csv')
    theta = torch.tensor(theta.iloc[:, 1:].to_numpy())
    f = torch.tensor(f.iloc[:, 1:].to_numpy()).T

    f = ((f.T - f.min(1).values) / (f.max(1).values - f.min(1).values)).T

    # arbitrary x
    m, n_all = f.shape
    x = np.arange(m)

    n = 800
    indtr = torch.randperm(n_all)[:n]
    indte = np.setdiff1d(np.arange(n_all), indtr)
    ftr = f[:, indtr]
    thetatr = theta[indtr]
    fte = f[:, indte]
    thetate = theta[indte]

    lr = 5e-4
    p = int(n * 1/4)

    psi = f.mean(1).unsqueeze(1)
    Phi, S, _ = torch.linalg.svd(ftr - psi, full_matrices=False)

    kap = 10
    Phi = Phi[:, :kap]
    S = S[:kap]
    Phi_mse = ((f - Phi @ Phi.T @ f) ** 2).mean()

    # with profile(activities=[ProfilerActivity.CPU],
    #              with_modules=True, record_shapes=True) as prof:
    model_ip_full = MVN_elbo_autolatent_sp(Phi=Phi, F=f, theta=theta, p=n)
    model_ip_full, _, _ = optim_elbo_lbfgs(model_ip_full, maxiter=100, lr=lr)
    rmse_ip_full = model_ip_full.test_rmse(theta0=thetate, f0=fte)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # prof.export_chrome_trace('trace_computeMV_detach_zerograd.json')
    model_ip4 = MVN_elbo_autolatent_sp(Phi=Phi, F=f, theta=theta, p=int(n/4))
    model_ip4, niter, flag = optim_elbo_lbfgs(model_ip4, maxiter=100, lr=lr)
    rmse_ip4 = model_ip4.test_rmse(theta0=thetate, f0=fte)

    import time
    surmise_t0 = time.time()
    surmise_emu = emulator(x=x, theta=thetatr.numpy(), f=ftr.numpy(), method='PCGPwM',
                           args={'standardpcinfo': {'U': Phi.numpy()}})
    surmise_t1 = time.time()
    surmise_pred = surmise_emu.predict(x=x, theta=thetate.numpy())
    rmse_surmise = np.sqrt(((surmise_pred.mean() - fte.numpy())**2).mean())
    print('surmise build time: {:.3f}'.format(surmise_t1 - surmise_t0))
    print('surmise rmse: {:.6f}'.format(rmse_surmise))