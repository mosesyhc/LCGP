import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)

from prediction import pred_gp
from likelihood import negloglik_mvlatent
from fayans_support import read_data, get_psi, get_Phi, read_only_complete_data


class MVlatentGP(nn.Module):
    def __init__(self, Lmb, G, sigma, theta, f, psi, Phi):
        super().__init__()
        self.Lmb = nn.Parameter(Lmb)
        self.G = nn.Parameter(G, requires_grad=False)
        self.sigma = nn.Parameter(sigma)
        self.theta = theta
        self.f = f
        self.psi = psi
        self.Phi = Phi
        self.kap = Phi.shape[1]

    def forward(self, theta0):
        Lmb = self.Lmb
        theta = self.theta
        G = self.G
        sigma = self.sigma

        psi = self.psi
        Phi = self.Phi

        kap = self.kap
        n0 = theta0.shape[0]

        Gpred = torch.zeros(n0, kap)
        for k in range(kap):
            Gpred[:, k], _ = pred_gp(lmb=Lmb[:, k], theta=theta, thetanew=theta0, g=G[:, k])
        fpred = (psi + Phi @ Gpred.T) / sigma

        return fpred

    def lik(self):
        Lmb = self.Lmb
        theta = self.theta
        G = self.G
        sigma = self.sigma

        f = self.f
        psi = self.psi
        Phi = self.Phi
        return negloglik_mvlatent(Lmb=Lmb, sigma=sigma, G=G, theta=theta, f=f, psi=psi, Phi=Phi)

    def test_mse(self, theta0, f0):
        Lmb = self.Lmb
        theta = self.theta
        G = self.G
        sigma = self.sigma

        kap = self.kap
        psi = self.psi
        Phi = self.Phi
        n0 = theta0.shape[0]

        Gpred = torch.zeros(n0, kap)
        for k in range(kap):
            Gpred[:, k], _ = pred_gp(lmb=Lmb[:, k], theta=theta, thetanew=theta0, g=G[:, k])
        fpred = (psi + Phi @ Gpred.T) / sigma

        return ((fpred - f0) ** 2).mean()


def test_mvlatent():
    f, x, theta = read_only_complete_data(r'../data/')

    f = torch.tensor(f)
    x = torch.tensor(x)
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

    from surmise.emulation import emulator
    emu = emulator(x=x.numpy(), theta=thetatr.numpy(),
                   f=ftr.numpy(), method='PCGPwM',
                   args={'warnings': True})

    emupred = emu.predict(x=x.numpy(), theta=thetate.numpy())
    emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    emutrainmse = ((emu.predict().mean() - ftr.numpy())**2).mean()
    print('surmise mse: {:.3f}'.format(emumse))
    print('surmise mse: {:.3f}'.format(emutrainmse))

    psi = ftr.mean(1).unsqueeze(1)
    Phi = get_Phi(x)
    kap = Phi.shape[1]
    d = theta.shape[1]

    # torch.manual_seed(1)

    Lmb = torch.Tensor(torch.randn(kap, d+1))
    # G = torch.Tensor(torch.randn(ntrain, kap))
    sigma = torch.Tensor((1,))
    G = (sigma * ftr - psi) @ torch.linalg.inv(Phi.T @ Phi)

    model = MVlatentGP(Lmb=Lmb, G=G, sigma=sigma,
                       theta=thetatr, f=ftr,
                       psi=psi, Phi=Phi)
    model.double()
    model.requires_grad_()

    lik = model.lik()

    lr = 10e-3
    # optim = torch.optim.LBFGS(model.parameters(), lr, line_search_fn='strong_wolfe')
    optim = torch.optim.Adam(model.parameters(), lr)  #, line_search_fn='strong_wolfe')

    header = ['iter', 'negloglik', 'test mse']
    print('{:<5s} {:<12s} {:<12s}'.format(*header))
    # print('{:<5s} {:<12.3f} {:<10.3f}'.format(' ', lik, model.test_mse(thetate, fte)))
    for epoch in range(500):
        optim.zero_grad()
        lik = model.lik()
        lik.backward()
        optim.step(lambda: model.lik())

        mse = model.test_mse(thetate, fte)
        trainmse = model.test_mse(thetatr, ftr)
        if epoch % 25 == 0:
            print('{:<5d} {:<12.3f} {:<10.3f}'.format(epoch, lik, trainmse))

#
if __name__ == '__main__':
    test_mvlatent()
