import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from fayans_support import read_data, read_test_data, read_only_complete_data
from mvn_elbo_autolatent_model import MVN_elbo_autolatent

from optim_elbo import optim_elbo, optim_elbo_lbfgs

import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_dir = r'code/data/borehole_data/'
    f, x0, theta = read_only_complete_data(data_dir)
    n_all = f.shape[1]

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)

    ftr = f[:, :200]
    thetatr = theta[:200]

    fmean = ftr.mean(1).unsqueeze(1)
    fstd = ftr.std(1).unsqueeze(1)

    ftr = (ftr - fmean) / fstd

    ### replication,
    kap = 1

    torch.manual_seed(0)

    n = ftr.shape[1]
    lr = 5e-4
    p = int(n * 1/4)

    Phi, _, _ = torch.linalg.svd(ftr, full_matrices=False)
    Phi = Phi[:, :kap]

    # Phi = torch.zeros()
    Phi_mse = ((ftr - Phi @ Phi.T @ ftr) ** 2).mean()

    # with profile(activities=[ProfilerActivity.CPU],
    #              with_modules=True, record_shapes=True) as prof:
    model = MVN_elbo_autolatent(Phi=Phi, F=ftr, theta=thetatr)

    model, _, _ = optim_elbo_lbfgs(model, maxiter=100, lr=lr)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # prof.export_chrome_trace('trace_computeMV_detach_zerograd.json')
    # model_mvgp = MVN_elbo_autolatent(Phi=Phi, F=f, theta=theta)
    # model_mvgp, niter, flag = optim_elbo_lbfgs(model_mvgp, maxiter=100, lr=lr)

    testind = torch.randperm(1800)[:500]+200
    thetate = theta[testind]
    fte = f[:, testind]
    fte = (fte - fmean) / fstd
    # with torch.no_grad():
    predmean = model.predictmean(thetate)
    predvar = model.predictvar(thetate)

    # GP at the 0th component
    from matern_covmat import cormat

    ck = cormat(thetatr, thetatr, model.lLmb[0])
    Ck = cormat(thetatr, thetatr, model.lLmb[0])

    Wk, Uk = torch.linalg.eigh(Ck)
    Ukh = Uk / torch.sqrt(Wk)

    Ckinvh_Vkh = Ukh * torch.sqrt(model.V[0])
    In = torch.eye(200)

    mu = ck @ Ukh @ (Ukh.T * model.M[0]).sum(1)
    tau2 = (ck - ck @ Ukh @ (In - Ukh * model.V[0] @ Ukh.T) @ Ukh.T @ ck).diag()
    tau2_tr = (ck - ck @ Ukh @ Ukh.T @ ck).diag()
    # tests
    with torch.no_grad():
        g0 = (Phi.T @ fte)
        ck0 = cormat(thetate, thetatr, model.lLmb[0])
        ck00 = cormat(thetate, thetate, model.lLmb[0], diag_only=True)
        mu0 = ck0 @ Ukh @ (Ukh.T * model.M[0]).sum(1)
        tau20 = ck00 - (ck0 @ Ukh @ (In - Ukh * model.V[0] @ Ukh.T) @ Ukh.T @ ck0.T).diag()

        tau2_new = ck00 - ck0 @ (Ukh @ Ukh.T @ ck0.T).diag()

    plt.hist((g0 - mu0) / tau20.sqrt())