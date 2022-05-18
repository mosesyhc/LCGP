import torch
import torch.nn as nn
from basis_nn_model import Basis
from optim_rules import convergence_f


def optim_Phi(F, kap, maxiter_nn=10000):
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
    Phi_as_param = Basis(m, kap, normalize=True)  #, inputdata=F)
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
