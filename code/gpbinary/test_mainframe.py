import torch
import torch.nn as nn
from fayans_support import read_data, read_test_data
from sklearn.cluster import KMeans
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp

dir = r'code/data/borehole_data/'
testf, testtheta = read_test_data(dir)

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


def optim_IPGPVI(F, Phi, Phi_loss, thetatr, thetai, maxiter_gp=1000):
    from torchviz import make_dot
    m, _ = F.shape

    lsigma2 = torch.Tensor(torch.log(Phi_loss))
    model = MVN_elbo_autolatent_sp(Lmb=None, initLmb=True,
                                   lsigma2=lsigma2, psi=torch.zeros(m),
                                   Phi=Phi, F=F, theta=thetatr, thetai=thetai)

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=1e-3)  # , line_search_fn='strong_wolfe')
    header = ['iter', 'neg elbo', 'train mse']
    epoch = 0
    print('\nELBO training:')
    print('{:<5s} {:<12s} {:<12s}'.format(*header))
    while True:
        optim.zero_grad()
        negelbo = model.negelbo()
        negelbo.backward(retain_graph=True)
        optim.step()  # lambda: model.lik())

        epoch += 1
        if epoch > maxiter_gp:
            break

        pred = model(thetatr)
        plt.plot((pred - F).detach().numpy())

        break
        # trainmse = model.test_mse(thetatr, F)

        # if epoch % 10 == 0:
        print('{:<5d} {:<12.6f}  {:<12.6f}'.format(epoch, negelbo, trainmse))

    return negelbo


# def test_sp(n_inducing, part, kap=5):
part = 1
f, x0, theta = read_data(dir)

f = torch.tensor(f)
x = torch.tensor(x0)
theta = torch.tensor(theta)

m, n = f.shape  # nloc, nparam

torch.manual_seed(0)
train_ind = torch.randperm(n)[((part - 1) * int(n / 10)):(part * int(n / 10))]
torch.seed()
ftr = f[:, train_ind]
thetatr = theta[train_ind]

# choose inducing points
n_inducing = 200
kmeans_theta = KMeans(n_clusters=n_inducing, algorithm='full').fit(thetatr)
thetai = torch.tensor(kmeans_theta.cluster_centers_)

psi = ftr.mean(1).unsqueeze(1)
F = ftr - psi
# Frng = torch.max(F, axis=0).values - torch.min(F, axis=0).values
# F /= Frng

kap = 5
Phi, Phi_loss = optim_Phi(F, kap)
# Phi, _, _ = torch.linalg.svd(F, full_matrices=False)
# Phi = Phi[:, :kap]
# Phi_loss = torch.mean((Phi @ Phi.T @ F - F)**2)
# print('Phi loss: {:.6f}'.format(Phi_loss))

negelbo = optim_IPGPVI(F=F, Phi=Phi, Phi_loss=Phi_loss, thetatr=thetatr, thetai=thetai, maxiter_gp=1000)

