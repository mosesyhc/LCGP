import torch
torch.set_default_dtype(torch.double)

from matern_covmat import covmat, cov_sp

from scipy.stats.qmc import LatinHypercube
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def diff(res, n, p, lsigma2):
    theta = sampler.random(n)
    kmeans = KMeans(p).fit(theta)

    thetai = kmeans.cluster_centers_
    d_theta = pairwise_distances(theta, thetai).mean()

    theta = torch.tensor(theta)
    thetai = torch.tensor(thetai)
    lsigma2 = torch.tensor(lsigma2)

    llmb = 0.5 * torch.log(torch.Tensor([theta.shape[1]])) + torch.log(torch.std(theta, 0))
    llmb = torch.cat((llmb, torch.Tensor([0])))

    Dd, Qh, logdetC = cov_sp(theta=theta, thetai=thetai, llmb=llmb)
    C = covmat(x1=theta, x2=theta, llmb=llmb)

    suppI = (Dd.diag() - Qh @ Qh.T) @ C
    d_mean_I = torch.abs(torch.eye(n) - suppI).median()
    d_max_I = torch.sqrt(((torch.eye(n) - suppI)**2).max())

    res['d_theta'] = d_theta
    res['d_mean_I'] = d_mean_I.item()
    res['d_max_I'] = d_max_I.item()
    return res


sampler = LatinHypercube(4)

res_struct = dict.fromkeys(['rep', 'n', 'p', 'ip_frac', 'd_theta', 'd_mean_I', 'd_max_I', 'lsigma2'])

ns = [200, 400, 800, 1600]
ip_fracs = [1, 1/2, 1/4, 1/8]
lsigma2s = [-12, -8, -4, -2, -1]

reslist = list()
for rep in range(5):
    for n in ns:
        for ip_frac in ip_fracs:
            for lsigma2 in lsigma2s:
                res = res_struct.copy()
                res['rep'] = rep
                res['n'] = n
                res['ip_frac'] = ip_frac
                p = int(n*ip_frac)
                res['p'] = p
                res['lsigma2'] = lsigma2

                res = diff(res, n, p, lsigma2)
                reslist.append(res)

import pandas as pd
df = pd.DataFrame(reslist)
df.to_csv('kmeans_trial.csv')


# import seaborn as sns
#
# thetadf = pd.DataFrame(theta)
# thetadf['label'] = 'original'
# thetaidf = pd.DataFrame(thetai)
# thetaidf['label'] = 'kmeans'
#
# df = pd.concat((thetadf, thetaidf), ignore_index=True)
#
# sns.pairplot(df, hue='label')