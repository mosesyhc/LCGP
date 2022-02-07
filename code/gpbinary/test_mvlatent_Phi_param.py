import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device(0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from prediction import pred_gp
from likelihood import negloglik_mvlatent
from fayans_support import read_data, get_psi, get_empPhi, read_only_complete_data

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class MVlatentGP(nn.Module):
    def __init__(self, Lmb, G, theta, f, psi, Phi):
        super().__init__()
        self.Lmb = nn.Parameter(Lmb)
        self.G = nn.Parameter(G, requires_grad=False)
        self.theta = theta
        self.f = f
        self.psi = psi
        self.Phi = Phi
        self.kap = Phi.shape[1]

    def forward(self, theta0):
        Lmb = self.Lmb
        theta = self.theta
        G = self.G

        psi = self.psi
        Phi = self.Phi

        kap = self.kap
        n0 = theta0.shape[0]

        Gpred = torch.zeros(n0, kap)
        for k in range(kap):
            Gpred[:, k], _ = pred_gp(lmb=Lmb[k], theta=theta, thetanew=theta0, g=G[:, k])
        fpred = (psi + Phi @ Gpred.T)

        return fpred

    def lik(self):
        Lmb = self.Lmb
        theta = self.theta
        G = self.G

        f = self.f
        psi = self.psi
        Phi = self.Phi
        return negloglik_mvlatent(Lmb=Lmb, G=G, theta=theta, f=f, psi=psi, Phi=Phi)

    def test_mse(self, theta0, f0):
        Lmb = self.Lmb
        theta = self.theta
        G = self.G
        kap = self.kap
        psi = self.psi
        Phi = self.Phi
        n0 = theta0.shape[0]

        Gpred = torch.zeros(n0, kap)
        for k in range(kap):
            Gpred[:, k], _ = pred_gp(lmb=Lmb[k], theta=theta, thetanew=theta0, g=G[:, k])
        fpred = (psi + Phi @ Gpred.T)

        return ((fpred - f0) ** 2).mean()


def get_bestPhi(f):
    U, S, V = torch.linalg.svd(f, full_matrices=False)

    return U


def returnPhi(Phi):
    return Phi


def test_mvlatent():
    f, x0, theta = read_only_complete_data(r'code/data/')

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)

    m, n = f.shape

    ntrain = 50
    ntest = 200

    tempind = torch.randperm(n)
    tr_inds = tempind[:ntrain]
    te_inds = tempind[-ntest:]

    ftr = f[:, tr_inds]
    thetatr = theta[tr_inds]
    fte = f[:, te_inds]
    thetate = theta[te_inds]

    # SURMISE BLOCK
    # # if False:
    # from surmise.emulation import emulator
    # emu = emulator(x=x.numpy(), theta=thetatr.numpy(),
    #                f=ftr.numpy(), method='PCGPwM',
    #                args={'warnings': True})
    #
    # emupred = emu.predict(x=x.numpy(), theta=thetate.numpy())
    # emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    # emutrainmse = ((emu.predict().mean() - ftr.numpy())**2).mean()
    # print('surmise mse: {:.3f}'.format(emumse))
    # print('surmise training mse: {:.3f}'.format(emutrainmse))

    psi = ftr.mean(1).unsqueeze(1)
    d = theta.shape[1]
    kap = 20

    def loss(get_Phi_, x, F):
        Phi = get_Phi_(x)
        kap = Phi.shape[1]

        # What can G be? Other than the LS solution.
        if hasattr(get_Phi_, 'gs'):
            if get_Phi_.gs:
                G = (Phi.T @ F).T
            else:
                G = (torch.linalg.solve(Phi.T @ Phi + 10e-8 * torch.eye(kap), Phi.T @ F)).T
        else:
            G = (torch.linalg.solve(Phi.T @ Phi + 10e-8 * torch.eye(kap), Phi.T @ F)).T

        mse = nn.MSELoss(reduction='mean')
        l = mse(Phi @ G.T, F)
        return l

    x = torch.column_stack((x0[:, 0], x0[:, 1],
                            *[x0[:, 2] == k for k in torch.unique(x0[:, 2])]))

    from basis_nn_model import Basis
    def loss_Phi_as_param_F(class_Phi, F):
        Phi = class_Phi()
        if class_Phi.normalize:
            Fhat = Phi @ Phi.T @ F
        else:
            Fhat = Phi @ torch.linalg.solve(Phi.T @ Phi, Phi.T) @ F
        mse = nn.MSELoss(reduction='mean')
        return mse(Fhat, F)


    # get_bestPhi takes F and gives the SVD U
    print('SVD error: ', loss(get_bestPhi, ftr - psi, ftr - psi))

    Phi0 = get_bestPhi(ftr - psi)

    Phi_as_param = Basis(m, kap, normalize=True)
    optim_nn = torch.optim.Adam(Phi_as_param.parameters(), lr=10e-3)


    nepoch_nn = 100
    print('Neural network training:')
    print('{:<5s} {:<12s}'.format('iter', 'train MSE'))
    for epoch in range(nepoch_nn):
        optim_nn.zero_grad()
        l = loss_Phi_as_param_F(Phi_as_param, ftr - psi)
        l.backward()
        optim_nn.step()
        if (epoch % 50 - 1) == 0:
            print('{:<5d} {:<12.6f}'.format(epoch, l))
    Phi_match = Phi_as_param().detach()

    if True:
        plt.style.use(['science', 'grid', 'no-latex'])
        plt.figure()
        # plt.hist((Phi0 @ Phi0.T @ (ftr - psi) - (ftr - psi)).reshape(-1).numpy(), bins=30, density=True, label='SVD')
        plt.hist((Phi_match @ torch.linalg.solve(Phi_match.T @ Phi_match,
                  Phi_match.T @ (ftr - psi)) - (ftr - psi)).reshape(-1).numpy(), bins=30,
                 fill=False, density=True, label='matchNN')
        plt.xlabel(r'prediction error')
        plt.ylabel(r'density')
        plt.tight_layout()
        plt.show()
    print('Reproducing Phi0 error in prediction of F: ', loss(returnPhi, Phi_match, ftr - psi))

    Phi = Phi_match
    G = (Phi.T @ (ftr - psi)).T

    # Phi @ G.T \approx Up @ W.T
    print('MSE between Phi @ G.T and (ftr - psi): {:.3f}'.format(torch.mean((Phi @ G.T - (ftr - psi))**2)))

    print('Basis size: ', Phi.shape)

    Lmb = torch.randn(kap, d+1)
    model = MVlatentGP(Lmb=Lmb, G=G,
                       theta=thetatr, f=ftr,
                       psi=psi, Phi=Phi)
    model.double()
    model.requires_grad_()

    ftrpred = model(thetatr)
    print('GP training MSE: {:.3f}'.format(torch.mean((ftr - ftrpred)**2)))
    # optim = torch.optim.LBFGS(model.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim = torch.optim.AdamW(model.parameters(), lr=10e-2)  #, line_search_fn='strong_wolfe')
    nepoch_gp = 100

    header = ['iter', 'negloglik', 'test mse', 'train mse']
    print('\nGP training:')
    print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(nepoch_gp):
        optim.zero_grad()
        lik = model.lik()
        lik.backward()
        optim.step(lambda: model.lik())

        mse = model.test_mse(thetate, fte)
        trainmse = model.test_mse(thetatr, ftr)
        # if epoch % 25 == 0:
        print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format(epoch, model.lik(), mse, trainmse))


if __name__ == '__main__':
    test_mvlatent()
