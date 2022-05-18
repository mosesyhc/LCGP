import torch
import torch.nn as nn
from fayans_support import read_data, read_test_data
from sklearn.cluster import KMeans
from optim_rules import convergence_f, convergence_g
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp

torch.set_default_dtype(torch.float64)

dir = r'code/data/borehole_data/'
testf, testtheta = read_test_data(dir)

f, x0, theta = read_data(dir)

f = torch.tensor(f)
x = torch.tensor(x0)
theta = torch.tensor(theta)

m, n = f.shape  # nloc, nparam

nepoch_nn = 20000
kap = 5

part = 1
torch.manual_seed(0)
train_ind = torch.randperm(theta.shape[0])[((part - 1) * int(n / 10)):(part * int(n / 10))]
torch.seed()
ftr = f[:, train_ind]

psi = ftr.mean(1).unsqueeze(1)
F = ftr - psi

from basis_nn_model import Basis
def loss(class_Phi, F):
    Phi = class_Phi()
    if class_Phi.normalize:
        Fhat = Phi @ Phi.T @ F
    else:
        Fhat = Phi @ torch.linalg.solve(Phi.T @ Phi, Phi.T) @ F
    mse = nn.MSELoss(reduction='mean')
    return mse(Fhat, F)


# get_bestPhi takes F and gives the SVD U
Phi_as_param = Basis(m, kap, normalize=True)
optim_nn = torch.optim.SGD(Phi_as_param.parameters(), lr=1e-2)
epoch = 0
l_prev = torch.inf

while True:
    optim_nn.zero_grad()
    l = loss(Phi_as_param, F)
    l.backward()
    optim_nn.step()

    epoch += 1
    if epoch > nepoch_nn:
        break
    elif convergence_f(l_prev, l):
        break

    l_prev = l.clone().detach()

print('loss: {:.3f} after {:d} iterations'.format(l, epoch))
Phi = Phi_as_param().detach()
