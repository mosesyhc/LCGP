import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])
import torch
from scipy.stats.qmc import LatinHypercube, Sobol
from sklearn.cluster import KMeans

from TestingfunctionBorehole import borehole_model
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp
from optim_elbo import optim_elbo

import pandas as pd


fig_dir = r'code/fig/choice_ip/'
res_dir = r'code/test_results/choice_ip/'


def accuracy_trial(rep, thetai, theta, ftr, thetate, fte, Phi, title):
    n = theta.shape[0]
    p = thetai.shape[0]
    ip_frac = p/n

    model = MVN_elbo_autolatent_sp(lLmb=None, initlLmb=True,
                               lsigma2=None, initlsigma2=True,
                               Phi=Phi, F=ftr, theta=theta, thetai=thetai)
    model, niter, flag = optim_elbo(model, ftr, theta, fte, thetate,
                                    maxiter=100, lr=lr)
    print('title: {:s}, p = {:d}'.format(title, p))
    print('test error', model.test_rmse(thetate, fte))
    rmse_by_theta = model.test_individual_error(thetate, fte)

    d = theta.shape[1]
    fig, ax = plt.subplots(d, d, figsize=(6, 6))
    for i in range(0, d):
        for j in range(0, d):
            if i != j and j >= i:
                ax[i][j].scatter(thetate[:, j], thetate[:, i], s=5,
                                c=rmse_by_theta, cmap='hot_r')
                ax[i][j].scatter(thetai[:, i], thetai[:, j],
                                marker='s', s=10,
                                color='black', alpha=0.75)
            else:
                ax[i][j].get_xaxis().set_visible(False)
                ax[i][j].get_yaxis().set_visible(False)
    plt.suptitle('{:s}, p = {:d}'.format(title, thetai.shape[0]))
    plt.tight_layout()
    plt.savefig(fig_dir + r'{:s}_n{:d}_p{:d}.png'.format(title, n, p), dpi=150)
    res = {'title': title,
           'rep': rep,
           'n': n, 'p': p,
           'ip_frac': ip_frac,
           'testrmse': model.test_rmse(thetate, fte).item(),
           'trainrmse': model.test_rmse(theta, ftr).item()}
    df = pd.DataFrame(res, index=[torch.randint(0, 10000, (1,)).item()])
    df.to_csv(res_dir + r'rep{:d}_{:s}_n{:d}_p{:d}.csv'.format(rep, title, n, p))


m = 25
n = 400
kap = 5
# ip_frac = 1/8
# p = int(n * ip_frac)

# fix x
sampler_x = LatinHypercube(d=2, seed=0)
x = sampler_x.random(25)
sampler_theta = LatinHypercube(d=4)

lr = 5e-4
for i in range(5):
    theta = torch.tensor(sampler_theta.random(n))

    F = torch.tensor(borehole_model(x, theta))
    ftr = ((F.T - F.mean(1)) / F.std(1)).T

    thetate = torch.tensor(sampler_theta.random(1000))
    fte = torch.tensor(borehole_model(x, thetate))
    fte = ((fte.T - F.mean(1)) / F.std(1)).T

    Phi, _, _ = torch.linalg.svd(ftr, full_matrices=False)
    Phi = Phi[:, :kap]
    for ip_frac in [1/8, 1/4, 1/2, 1]:
        p = int(n * ip_frac)

        kmeans_theta = KMeans(n_clusters=p, algorithm='full').fit(theta)
        thetai = torch.tensor(kmeans_theta.cluster_centers_)
        ind2 = torch.argsort(theta[:, 2])
        thetai_art = torch.row_stack((thetai[10:].clone(), theta[ind2][:10].clone()))
        thetai_lhs = torch.tensor(sampler_theta.random(p))

        sampler_sobol = Sobol(d=4)
        thetai_sobol = torch.tensor(sampler_sobol.random(p))

        accuracy_trial(i, thetai, theta, ftr, thetate, fte, Phi, 'kmeans')
        # accuracy_trial(i, thetai_art, theta, ftr, thetate, fte, Phi, 'kmeans_w_art')
        # accuracy_trial(i, thetai_lhs, theta, ftr, thetate, fte, Phi, 'LHS')
        # accuracy_trial(i, thetai_sobol, theta, ftr, thetate, fte, Phi, 'Sobol')
    break
        # print(ip_frac)
