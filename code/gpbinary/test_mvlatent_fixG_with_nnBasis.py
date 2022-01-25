import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)

from prediction import pred_gp
from likelihood import negloglik_mvlatent
from fayans_support import read_data, get_psi, get_empPhi, read_only_complete_data


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
            Gpred[:, k], _ = pred_gp(lmb=Lmb[:, k], theta=theta, thetanew=theta0, g=G[:, k])
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


def test_mvlatent():
    f, x, theta = read_only_complete_data(r'code/data/')

    f = torch.tensor(f)
    x = torch.tensor(x)

    theta = torch.tensor(theta)
    m, n = f.shape

    ntrain = 50
    ntest = 200

    # torch.manual_seed(1)

    tempind = torch.randperm(n)
    tr_inds = tempind[:ntrain]
    te_inds = tempind[-ntest:]

    ftr = f[:, tr_inds]
    thetatr = theta[tr_inds]
    fte = f[:, te_inds]
    thetate = theta[te_inds]

    # if False:
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
    # Phi = get_Phi(x)
    # print(Phi.shape)
    # kap = Phi.shape[1]
    d = theta.shape[1]


    # G = (torch.linalg.solve(Phi.T  @ Phi + 10e-8 * torch.eye(kap), Phi.T @ (sigma * ftr - psi))).T
    # F = (Phi @ G.T + psi) / sigma

    def loss(get_Phi_, x, F):
        Phi = get_Phi_(x)
        kap = Phi.shape[1]

        G = (torch.linalg.solve(Phi.T @ Phi + 10e-8 * torch.eye(kap), Phi.T @ F)).T
        mse = nn.MSELoss(reduction='mean')
        l = mse(Phi @ G.T, F)
        return l

    x_cat = torch.column_stack((x[:, 0], x[:, 1],
                                x[:, 2] == 0,
                                x[:, 2] == 1,
                                x[:, 2] == 2,
                                x[:, 2] == 3,
                                x[:, 2] == 4,
                                x[:, 2] == 5,
                                x[:, 2] == 6,
                                x[:, 2] == 7,
                                x[:, 2] == 8, ))

    l0 = loss(get_empPhi, x, ftr - psi)

    from basis_nn_model import BasisGenNNType
    kap = 100
    get_Phi_ = BasisGenNNType(kap)
    get_Phi_.double()

    # optim_nn = torch.optim.LBFGS(get_Phi_.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim_nn = torch.optim.Adam(get_Phi_.parameters(), lr=10e-4)
    print('{:<5s} {:<12s} {:<12s}'.format('iter', 'MSE', 'baseline MSE'))
    for epoch in range(300):
        optim_nn.zero_grad()
        l = loss(get_Phi_, x_cat, ftr - psi)
        l.backward()
        optim_nn.step()

        if (epoch % 100 - 1) == 0:
            print('{:<5d} {:<12.3f} {:<12.3f}'.format(epoch, l, l0))

    Phi = get_Phi_(x_cat).detach()
    U, S, V = torch.linalg.svd(Phi, full_matrices=False)

    U, W = torch.linalg.eig(Phi.T @ Phi)

    raise

    ### Gaussian process
    Phi = get_Phi_(x).detach()
    G = (torch.linalg.solve(Phi.T @ Phi + 10e-8 * torch.eye(kap), Phi.T @ (ftr - psi))).T
    Lmb = torch.Tensor(torch.randn(kap, d+1))

    print('Basis size: ', Phi.shape)

    model = MVlatentGP(Lmb=Lmb, G=G,
                       theta=thetatr, f=ftr,
                       psi=psi, Phi=Phi)
    model.double()
    model.requires_grad_()

    # optim = torch.optim.LBFGS(model.parameters(), lr, line_search_fn='strong_wolfe')
    optim = torch.optim.Adam(model.parameters(), lr=10e-3)  #, line_search_fn='strong_wolfe')

    header = ['iter', 'negloglik', 'test mse', 'train mse']
    print('\n{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(10):
        optim.zero_grad()
        lik = model.lik()
        lik.backward()
        optim.step(lambda: model.lik())

        mse = model.test_mse(thetate, fte)
        trainmse = model.test_mse(thetatr, ftr)
        # if epoch % 25 == 0:
        print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format(epoch, lik, mse, trainmse))

#
if __name__ == '__main__':
    test_mvlatent()
