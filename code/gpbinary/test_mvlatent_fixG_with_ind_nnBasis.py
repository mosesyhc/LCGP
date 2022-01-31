import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device(0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def test_mvlatent():
    f, x0, theta = read_only_complete_data(r'code/data/')

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)

    # f = f.to(device)
    # x = x.to(device)
    # theta = theta.to(device)

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
    #
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
    # Phi = get_Phi(x)
    # print(Phi.shape)
    # kap = Phi.shape[1]
    d = theta.shape[1]

    # G = (torch.linalg.solve(Phi.T  @ Phi + 10e-8 * torch.eye(kap), Phi.T @ (sigma * ftr - psi))).T
    # F = (Phi @ G.T + psi) / sigma

    def loss(get_Phi_, x, F):
        Phi = get_Phi_(x)
        kap = Phi.shape[1]

        # What can G be? Other than the LS solution.
        G = (torch.linalg.solve(Phi.T @ Phi + 10e-8 * torch.eye(kap), Phi.T @ F)).T
        mse = nn.MSELoss(reduction='mean')
        l = mse(Phi @ G.T, F)
        return l

    x = torch.column_stack((x0[:, 0], x0[:, 1],
                            *[x0[:, 2] == k for k in torch.unique(x0[:, 2])]))

    l0 = loss(get_empPhi, x0, ftr - psi)

    from basis_nn_model import BasisGenNNTypeMulti
    kap = 50
    # x.cuda()

    get_Phi_ = BasisGenNNTypeMulti(kap, x)
    # get_Phi_.cuda()
    get_Phi_.double()

    # Neural network to find basis
    # optim_nn = torch.optim.LBFGS(get_Phi_.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim_nn = torch.optim.Adam(get_Phi_.parameters(), lr=10e-4)
    print('Neural network training:')
    print('{:<5s} {:<12s} {:<12s}'.format('iter', 'MSE', 'baseline MSE'))
    for epoch in range(100):
        optim_nn.zero_grad()
        l = loss(get_Phi_, x, ftr - psi)
        l.backward()
        optim_nn.step()
        if (epoch % 25 - 1) == 0:
            print('{:<5d} {:<12.3f} {:<12.3f}'.format(epoch, l, l0))
    Phi = get_Phi_(x).detach()

    # further reduce rank
    U, S, V = torch.linalg.svd(Phi, full_matrices=False)
    # ind = (S / S.norm()) > 10e-7

    G = (torch.linalg.solve(Phi.T @ Phi + 10e-8 * torch.eye(kap), Phi.T @ (ftr - psi))).T
    W = (S.diag() @ V @ G.T).detach().T

    mse0 = torch.mean((Phi @ G.T - (ftr - psi))**2)
    print('{:<10s} {:<16s}'.format('no. basis', '(msep - mse0) / mse0'))
    for kap0 in torch.arange(10, kap+1, 10).to(int):
        Up = U[:, :kap0]  # keep
        Sp = S[:kap0]
        Vp = V[:kap0]

        Wp = (Sp.diag() @ Vp @ G.T).detach().T
        msep = torch.mean((Up @ Wp.T - (ftr - psi))**2)
        relmse = (msep - mse0) / mse0 / ((ftr - psi)**2).mean()
        print('{:<10d} {:<16.6f}'.format(kap0, relmse))
        if relmse < 0.01:
            break


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

#
if __name__ == '__main__':
    test_mvlatent()
