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

    ntrain = 10
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
    kap = ntrain

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

    l0 = loss(get_empPhi, x0, ftr - psi)

    from basis_nn_model import BasisGenNNTypeMulti

    # reproducing bestPhi with NN
    def loss_Phi(get_Phi_, Phi0):
        Phi = get_Phi_(x)
        mse = nn.MSELoss(reduction='mean')
        return mse(Phi, Phi0)

    matchPhi_ = BasisGenNNTypeMulti(kap, x, normalize=True)
    matchPhi_.double()


    from basis_nn_model import Phi_Class
    def loss_Phi_as_param(class_Phi, Phi0):
        Phi = class_Phi()
        mse = nn.MSELoss(reduction='mean')
        return mse(Phi, Phi0)

    def loss_Phi_as_param_F(class_Phi, F):
        Phi = class_Phi()
        Fhat = Phi @ torch.linalg.solve(Phi.T @ Phi, Phi.T) @ F
        mse = nn.MSELoss(reduction='mean')
        return mse(Fhat, F)





    # get_bestPhi takes F and gives the SVD U
    print('SVD error: ', loss(get_bestPhi, ftr - psi, ftr - psi))

    Phi0 = get_bestPhi(ftr - psi)
    #
    #
    # # Neural network to match Phi0
    # print('Neural network training:')
    # print('{:<5s} {:<12s}'.format('iter', 'MSE bet. Phi'))
    # for epoch in range(200):
    #     optim_nn.zero_grad()
    #     l = loss_Phi(matchPhi_, x, Phi0)
    #     l.backward()
    #     optim_nn.step()
    #     if (epoch % 50 - 1) == 0:
    #         print('{:<5d} {:<12.6f}'.format(epoch, l))
    # Phi_match = matchPhi_(x).detach()

    Phi_as_param = Phi_Class(m, kap)
    optim_nn = torch.optim.Adam(Phi_as_param.parameters(), lr=10e-3)


    print('Neural network training:')
    print('{:<5s} {:<12s}'.format('iter', 'MSE bet. Phi'))
    for epoch in range(250):
        optim_nn.zero_grad()
        l = loss_Phi_as_param_F(Phi_as_param, ftr - psi)
        l.backward()
        optim_nn.step()
        if (epoch % 50 - 1) == 0:
            print('{:<5d} {:<12.6f}'.format(epoch, l))
    Phi_match = Phi_as_param().detach()

    plt.figure()
    plt.hist((Phi0 - Phi_match).reshape(-1).numpy(), bins=30, density=True)
    plt.xlabel(r'$\Phi - \Phi_0$')
    plt.ylabel(r'density')
    plt.show()

    plt.figure()
    # plt.hist((Phi0 @ Phi0.T @ (ftr - psi) - (ftr - psi)).reshape(-1).numpy(), bins=30, density=True, label='SVD')
    plt.hist((Phi_match @ torch.linalg.solve(Phi_match.T @ Phi_match,
              Phi_match.T @ (ftr - psi)) - (ftr - psi)).reshape(-1).numpy(), bins=30, density=True, label='matchNN')
    plt.xlabel(r'prediction error')
    plt.ylabel(r'density')
    plt.show()
    print('error bet. best Phi and reproduced Phi:', torch.mean((Phi0 - Phi_match)**2))
    print('Reproducing Phi0 error in prediction of F: ', loss(returnPhi, Phi_match, ftr - psi))

    raise

    # x.cuda()

    nn_lr = 5*10e-5
    nepoch = 250
    mseresults_w_gs = torch.zeros(nepoch)

    get_Phi_ = BasisGenNNTypeMulti(kap, x, normalize=True)
    # get_Phi_.cuda()
    get_Phi_.double()

    # Neural network to find basis
    # optim_nn = torch.optim.LBFGS(get_Phi_.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim_nn = torch.optim.Adam(get_Phi_.parameters(), lr=nn_lr)
    print('Neural network training:')
    print('{:<5s} {:<12s} {:<12s}'.format('iter', 'MSE', 'baseline MSE'))
    for epoch in range(nepoch):
        optim_nn.zero_grad()
        l = loss(get_Phi_, x, ftr - psi)
        l.backward()
        optim_nn.step()
        if (epoch % 25 - 1) == 0:
            print('{:<5d} {:<12.3f} {:<12.3f}'.format(epoch, l, l0))
        mseresults_w_gs[epoch] = l
    Phi = get_Phi_(x).detach()
    print(((Phi.T @ Phi - torch.eye(Phi.shape[1]))**2).mean())
    # further reduce rank
    U, S, V = torch.linalg.svd(Phi, full_matrices=False)
    # ind = (S / S.norm()) > 10e-7

    G = (Phi.T @ (ftr - psi)).T
    W = (S.diag() @ V @ G.T).detach().T

    relmses_gs = torch.zeros(kap)
    mse0 = torch.mean((Phi @ G.T - (ftr - psi))**2)
    print('{:<10s} {:<16s}'.format('no. basis', '(msep - mse0) / mse0'))
    for kap0 in torch.arange(1, kap+1).to(int):
        Up = U[:, :kap0]  # keep
        Sp = S[:kap0]
        Vp = V[:kap0]

        Wp = (Sp.diag() @ Vp @ G.T).detach().T
        msep = torch.mean((Up @ Wp.T - (ftr - psi))**2)
        relmse = (msep - mse0) / mse0 / ((ftr - psi)**2).mean()
        if kap0 % 10 == 0:
            print('{:<10d} {:<16.6f}'.format(kap0, relmse))
        # if relmse < 0.01:
        #     break
        relmses_gs[kap0-1] = relmse

    del get_Phi_, Phi, U, S, V, Up, Sp, Vp, G, W, Wp


    ############################################################
    mseresults_wo_gs = torch.zeros(nepoch)

    get_Phi_woGS = BasisGenNNTypeMulti(kap, x, normalize=False)
    # get_Phi_.cuda()
    get_Phi_woGS.double()

    # Neural network to find basis
    # optim_nn = torch.optim.LBFGS(get_Phi_.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim_nn = torch.optim.Adam(get_Phi_woGS.parameters(), lr=nn_lr)
    print('Neural network training:')
    print('{:<5s} {:<12s} {:<12s}'.format('iter', 'MSE', 'baseline MSE'))
    for epoch in range(nepoch):
        optim_nn.zero_grad()
        l = loss(get_Phi_woGS, x, ftr - psi)
        l.backward()
        optim_nn.step()
        if (epoch % 25 - 1) == 0:
            print('{:<5d} {:<12.3f} {:<12.3f}'.format(epoch, l, l0))
        mseresults_wo_gs[epoch] = l
    Phi2 = get_Phi_woGS(x).detach()

    print(((Phi2.T @ Phi2 - torch.eye(Phi2.shape[1]))**2).mean())
    # further reduce rank
    U, S, V = torch.linalg.svd(Phi2, full_matrices=False)
    # ind = (S / S.norm()) > 10e-7

    G = (torch.linalg.solve(Phi2.T @ Phi2 + 10e-8 * torch.eye(kap), Phi2.T @ (ftr - psi))).T
    W = (S.diag() @ V @ G.T).detach().T

    relmses_wo_gs = torch.zeros(kap)
    mse0 = torch.mean((Phi2 @ G.T - (ftr - psi))**2)
    print('{:<10s} {:<16s}'.format('no. basis', '(msep - mse0) / mse0'))
    for kap0 in torch.arange(1, kap+1).to(int):
        Up = U[:, :kap0]  # keep
        Sp = S[:kap0]
        Vp = V[:kap0]

        Wp = (Sp.diag() @ Vp @ G.T).detach().T
        msep = torch.mean((Up @ Wp.T - (ftr - psi))**2)
        relmse = (msep - mse0) / mse0 / ((ftr - psi)**2).mean()
        if kap0 % 10 == 0:
            print('{:<10d} {:<16.6f}'.format(kap0, relmse))
        # if relmse < 0.01:
        #     break
        relmses_wo_gs[kap0-1] = relmse
    ############################################################

    import numpy as np
    epochres = torch.column_stack((mseresults_w_gs, mseresults_wo_gs))
    relmseres = torch.column_stack((relmses_gs, relmses_wo_gs))
    np.savetxt('epoch_mse.txt', epochres.detach().numpy())
    np.savetxt('rel_testmse.txt', relmseres.detach().numpy())

    plt.figure()
    plt.plot(mseresults_w_gs.detach().numpy(), label='with GS')
    plt.plot(mseresults_wo_gs.detach().numpy(), label='without GS')
    plt.ylabel('train MSE')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(relmses_gs.detach().numpy(), label='with GS')
    plt.plot(relmses_wo_gs.detach().numpy(), label='without GS')
    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'relative MSE')
    plt.legend()
    plt.show()

    raise

    print('Truncation error in basis: {:.3E}'.format(((Phi - Up @ Sp.diag() @ Vp)**2).mean()))
    # Phi @ G.T \approx Up @ W.T
    print('MSE between Phi @ G.T and (ftr - psi): {:.3f}'.format(torch.mean((Phi @ G.T - (ftr - psi))**2)))
    print('MSE between full U @ W.T and (ftr - psi): {:.3f}'.format(torch.mean((U @ W.T - (ftr - psi))**2)))
    print('MSE between Up @ Wp.T and (ftr - psi): {:.3f}'.format(torch.mean((Up @ Wp.T - (ftr - psi))**2)))
    Lmb = torch.Tensor(torch.randn(kap0, d+1))

    print('Basis size: ', Up.shape)

    model = MVlatentGP(Lmb=Lmb, G=Wp,
                       theta=thetatr, f=ftr,
                       psi=psi, Phi=Up)
    model.double()
    model.requires_grad_()

    ftrpred = model(thetatr)
    print('GP training MSE: {:.3f}'.format(torch.mean((ftr - ftrpred)**2)))
    # optim = torch.optim.LBFGS(model.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim = torch.optim.AdamW(model.parameters(), lr=10e-2)  #, line_search_fn='strong_wolfe')

    header = ['iter', 'negloglik', 'test mse', 'train mse']
    print('\nGP training:')
    print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(25):
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
