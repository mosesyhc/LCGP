import torch
from torch import nn
from mvn_elbo_model import MVN_elbo
from fayans_support import read_only_complete_data
torch.set_default_dtype(torch.double)
import matplotlib.pyplot as plt
import numpy as np
torch.autograd.set_detect_anomaly(True)


nepoch_nn = 100
nepoch_elbo = 250

f, x0, theta = read_only_complete_data(r'code/data/')

f = torch.tensor(f)
x0 = torch.tensor(x0)
theta = torch.tensor(theta)

m, n = f.shape  # nloc, nparam

ntrain = 50
ntest = 200

torch.manual_seed(0)
tempind = torch.randperm(n)
tr_inds = tempind[:ntrain]
te_inds = tempind[-ntest:]
torch.seed()
ftr = f[:, tr_inds]
thetatr = theta[tr_inds]
fte = f[:, te_inds]
thetate = theta[te_inds]

psi = ftr.mean(1).unsqueeze(1)
d = theta.shape[1]
kap = 20

x = torch.column_stack((x0[:, 0], x0[:, 1],
                        *[x0[:, 2] == k for k in torch.unique(x0[:, 2])]))
F = ftr - psi

if False:
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
    optim_nn = torch.optim.Adam(Phi_as_param.parameters(), lr=10e-3)

    print('F shape:', F.shape)

    print('Neural network training:')
    print('{:<5s} {:<12s}'.format('iter', 'train MSE'))
    for epoch in range(nepoch_nn):
        optim_nn.zero_grad()
        l = loss(Phi_as_param, F)
        l.backward()
        optim_nn.step()
        if (epoch % 50 - 1) == 0:
            print('{:<5d} {:<12.6f}'.format(epoch, l))
    Phi_match = Phi_as_param().detach()

    print('Reproducing Phi0 error in prediction of F: ',
          torch.mean((Phi_match @ Phi_match.T @ F - F) ** 2))

    Phi = Phi_match
    print('Basis size: ', Phi.shape)
    np.savetxt('Phi_seed0.txt', Phi.numpy())

Phi = torch.tensor(np.loadtxt('Phi_seed0.txt'))

Lmb = torch.zeros(kap, d + 1)
Mu = Phi.T @ F
V = torch.ones((kap, ntrain)) * 0.01
lsigma2 = torch.Tensor([0])
model = MVN_elbo(Mu=Mu, V=V, Lmb=Lmb,
                 lsigma2=lsigma2, psi=torch.zeros_like(psi),
                 Phi=Phi, F=F, theta=thetatr)
model.double()

# # .requires_grad_() turns all parameters on.
# model.requires_grad_()
# print(list(model.parameters()))

ftrpred = model(thetatr)
print('ELBO model training MSE: {:.3f}'.format(torch.mean((F - ftrpred) ** 2)))
# optim = torch.optim.LBFGS(model.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=10e-3)  # , line_search_fn='strong_wolfe')
header = ['iter', 'neg elbo', 'test mse', 'train mse']
print('\nELBO training:')
print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
for epoch in range(nepoch_elbo):
    optim.zero_grad()
    negelbo = model.negelbo()
    negelbo.backward()
    optim.step()  # lambda: model.lik())

    mse = model.test_mse(thetate, fte - psi)
    trainmse = model.test_mse(thetatr, F)
    # if epoch % 25 == 0:
    print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format
          (epoch, negelbo, mse, trainmse))
