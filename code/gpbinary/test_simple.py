import torch
from torch.profiler import profile, record_function, ProfilerActivity
from fayans_support import read_data, read_test_data, read_only_complete_data
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from mvn_elbo_autolatent_sp_model_jit import MVN_elbo_autolatent_sp

from optim_elbo import optim_elbo, optim_elbo_lbfgs


if __name__ == '__main__':
    data_dir = r'code/data/borehole_data/'
    f, x0, theta = read_only_complete_data(data_dir)
    n_all = f.shape[1]

    f = torch.tensor(f)[:, :800]
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)[:800]

    fmean = f.mean(1).unsqueeze(1)
    fstd = f.std(1).unsqueeze(1)

    f = (f - fmean) / fstd

    ### replication,
    kap = 2

    n = f.shape[1]
    lr = 5e-4
    p = int(n * 1/4)

    Phi, _, _ = torch.linalg.svd(f, full_matrices=False)
    Phi = Phi[:, :kap]
    Phi_mse = ((f - Phi @ Phi.T @ f) ** 2).mean()

    model_ip_full = MVN_elbo_autolatent_sp(Phi=Phi, F=f, theta=theta, p=n)

    with profile(activities=[ProfilerActivity.CPU], with_modules=True, record_shapes=True) as prof:
        model_ip_full, _, _ = optim_elbo_lbfgs(model_ip_full, maxiter=100, lr=lr)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace('trace_kap2_ls2.json')

    # model_mvgp = MVN_elbo_autolatent(Phi=Phi, F=f, theta=theta)
    # model_mvgp, niter, flag = optim_elbo_lbfgs(model_mvgp, maxiter=100, lr=lr)
