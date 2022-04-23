import torch
import torch.nn as nn
from fayans_support import read_data, read_test_data
from sklearn.cluster import KMeans
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp
torch.set_default_dtype(torch.double)
torch.autograd.set_detect_anomaly(True)
from optim_rules import convergence_f, convergence_g
import matplotlib.pyplot as plt

def optim_Phi(F, kap, maxiter_nn=10000):
    from basis_nn_model import Basis
    from optim_rules import convergence_f

    m, _ = F.shape

    def loss(class_Phi, F):
        Phi = class_Phi()
        if class_Phi.normalize:
            Fhat = Phi @ Phi.T @ F
        else:
            Fhat = Phi @ torch.linalg.solve(Phi.T @ Phi, Phi.T) @ F
        mse = nn.MSELoss(reduction='mean')
        return mse(Fhat, F)

    # get_bestPhi takes F and gives the SVD U
    Phi_as_param = Basis(m, kap, normalize=True) #, inputdata=F)
    optim_nn = torch.optim.SGD(Phi_as_param.parameters(), lr=1e-2)
    epoch = 0
    l_prev = torch.inf
    while True:
        optim_nn.zero_grad()
        l = loss(Phi_as_param, F)
        l.backward()
        optim_nn.step()

        epoch += 1
        if epoch > maxiter_nn:
            break
        elif convergence_f(l_prev, l):
            break

        l_prev = l.detach()

    print('Phi loss: {:.3f} after {:d} iterations'.format(l, epoch))
    return Phi_as_param().detach(), l.detach()


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
n_inducing = int(thetatr.shape[0]/ 2)
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
Phi, Phi_loss = optim_Phi(F, kap)
# Phi, _, _ = torch.linalg.svd(F, full_matrices=False)
# Phi = Phi[:, :kap]
# Phi_loss = torch.mean((Phi @ Phi.T @ F - F)**2)
print('Phi loss:', Phi_loss)
m, _ = F.shape

# lsigma2 = torch.Tensor(torch.log(Phi_loss)) # torch.max(torch.Tensor((1e-3,)), )
model = MVN_elbo_autolatent_sp(Lmb=None, initLmb=True,
                               lsigma2=None, initsigma2=True,
                               # psi=torch.zeros_like(psi),
                               Phi=Phi, F=F, theta=thetatr, thetai=thetai)
model.double()

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=1e-3)  # , line_search_fn='strong_wolfe')
header = ['iter', 'neg elbo', 'mse', 'train mse']
epoch = 0
negelbo_prev = torch.inf

import numpy as np
import time
save_time = np.zeros(51)
save_time[0] = time.time()
print('\nn = {:d}, p = {:d}'.format(thetatr.shape[0], thetai.shape[0]))

print('\nELBO training:')
print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
while True:
    optim.zero_grad(set_to_none=True)
    negelbo = model.negelbo()
    negelbo.backward() #retain_graph=True)
    # from torchviz import make_dot
    # make_dot(negelbo, params=dict(list(model.named_parameters())),
    #          show_attrs=True, show_saved=True).render('negelbo_cg_retainTrue', format='png')

    optim.step()  # lambda: model.lik())

    if epoch % 10 == 0:
        with torch.no_grad():
            model.create_MV()
            trainmse = model.test_mse(thetatr, F)
            mse = model.test_mse(thetate, fte)

            print('{:<5d} {:<12.3f} {:<12.6f} {:<12.6f}'.format
                  (epoch, negelbo, mse, trainmse))
    elif convergence_f(negelbo_prev, negelbo):
        break

    epoch += 1
    save_time[epoch] = time.time()
    if epoch >= 50:
        break
    negelbo_prev = negelbo.clone().detach()


# np.savetxt('time_ip_{:d}.txt'.format(thetai.shape[0]), save_time)
# negelbo = optim_IPGPVI(F=F, Phi=Phi, Phi_loss=Phi_loss, psi=psi,
#                        thetatr=thetatr, thetai=thetai, maxiter_gp=1000,
#                        fte=fte, thetate=thetate)



# def optim_IPGPVI(F, Phi, Phi_loss, psi, thetatr, thetai, fte, thetate, maxiter_gp=1000):
#     m, _ = F.shape
#
#     # lsigma2 = torch.Tensor(torch.log(Phi_loss)) # torch.max(torch.Tensor((1e-3,)), )
#     model = MVN_elbo_autolatent_sp(Lmb=None, initLmb=True,
#                                    lsigma2=None, initsigma2=True,
#                                    # psi=torch.zeros_like(psi),
#                                    Phi=Phi, F=F, theta=thetatr, thetai=thetai)
#     model.double()
#
#     optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
#                              lr=1e-3)  # , line_search_fn='strong_wolfe')
#     header = ['iter', 'neg elbo', 'mse', 'train mse']
#     epoch = 0
#     print('\nELBO training:')
#     print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
#     while True:
#         optim.zero_grad()
#         negelbo = model.negelbo()
#         negelbo.backward(retain_graph=True)
#         optim.step()  # lambda: model.lik())
#
#         epoch += 1
#         if epoch > maxiter_gp:
#             break
#
#         # pred = model(thetatr)
#         trainmse = model.test_mse(thetatr, F)
#         mse = model.test_mse(thetate, fte - psi)
#
#         # if epoch % 10 == 0:
#         print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format
#               (epoch, negelbo, mse, trainmse))
#     return negelbo