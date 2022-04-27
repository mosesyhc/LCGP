import torch
import torch.nn as nn
from optim_basis_Phi import optim_Phi

from fayans_support import read_data, read_test_data
from sklearn.cluster import KMeans
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp
torch.set_default_dtype(torch.double)
torch.autograd.set_detect_anomaly(True)

import optim_rules
from optim_rules import convergence_f, convergence_g
import matplotlib.pyplot as plt

# def test_sp(n_inducing, part, kap=5):
part = 1
dir = r'code/data/borehole_data/'
f, x0, theta = read_data(dir)
fte, thetate = read_test_data(dir)


f = torch.tensor(f)
x = torch.tensor(x0)
theta = torch.tensor(theta)

fte = torch.tensor(fte)
thetate = torch.tensor(thetate)

m, n = f.shape  # nloc, nparam

torch.manual_seed(0)
train_ind = torch.randperm(n)[((part - 1) * int(n / 5)):(part * int(n / 5))]
torch.seed()
ftr = f[:, train_ind]
thetatr = theta[train_ind]

# choose inducing points
n_inducing = int(thetatr.shape[0] / 3)
kmeans_theta = KMeans(n_clusters=n_inducing, algorithm='full').fit(thetatr)
thetai = torch.tensor(kmeans_theta.cluster_centers_)

# thetai = thetatr.clone()
psi = ftr.mean(1).unsqueeze(1)

# F = ftr - psi
F = (ftr - psi) / ftr.std(1).unsqueeze(1)
fte = (fte - psi) / ftr.std(1).unsqueeze(1)

# Frng = torch.max(F, axis=0).values - torch.min(F, axis=0).values
# F /= Frng

kap = 4
print('optimizing Phi ... ')
Phi, Phi_loss = optim_Phi(F, kap)
# Phi, _, _ = torch.linalg.svd(F, full_matrices=False)
# Phi = Phi[:, :kap]
# Phi_loss = torch.mean((Phi @ Phi.T @ F - F)**2)
print('Phi loss:', Phi_loss)
m, _ = F.shape

# lsigma2 = torch.Tensor(torch.log(Phi_loss)) # torch.max(torch.Tensor((1e-3,)), )
model = MVN_elbo_autolatent_sp(lLmb=None, initlLmb=True,
                               lsigma2=None, initlsigma2=True,
                               Phi=Phi, F=F, theta=thetatr, thetai=thetai)
model.double()

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=8e-3)  # , line_search_fn='strong_wolfe')
header = ['iter', 'neg elbo', 'mse', 'train mse']
epoch = 0
negelbo_prev = torch.inf

import numpy as np
import time
# save_time = np.zeros(51)
# save_time[0] = time.time()
print('\nn = {:d}, p = {:d}'.format(thetatr.shape[0], thetai.shape[0]))

print('\nELBO training:')
print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
while True:
    optim.zero_grad(set_to_none=True)
    negelbo = model.negelbo()
    negelbo.backward()
    optim.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            model.create_MV()
            trainmse = model.test_mse(thetatr, F)
            mse = model.test_mse(thetate, fte)

            print('{:<5d} {:<12.3f} {:<12.6f} {:<12.6f}'.format
                  (epoch, negelbo, mse, trainmse))
        # save_time[epoch // 100] = time.time()
    elif convergence_f(negelbo_prev, negelbo):
        print('FTOL <= {:.3E}'.format(optim_rules.FTOL))
        break

    elif convergence_g(model.parameters()):
        print('GTOL <= {:.3E}'.format(optim_rules.GTOL))
        break

    elif epoch >= 1000:
        break

    epoch += 1
    negelbo_prev = negelbo.clone().detach()

# print('{:.3f} seconds after {:d} iterations'.format(time.time() - save_time[0], epoch))
# print(save_time)
#
# import matplotlib.pyplot as plt
# plt.plot(save_time - save_time[0])
# plt.show()