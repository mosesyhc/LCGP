import torch
import torch.nn as nn
from prediction import pred_gp
from likelihood import negloglik_gp
from matern_covmat import covmat


class MVlatentGP(nn.Module):
    def __init__(self, Lmb, G, Phi, lsigma, theta, f, psi,
                 optim_param_set=None):
        super().__init__()
        self.Lmb = nn.Parameter(Lmb)
        self.G = nn.Parameter(G)
        self.Phi = nn.Parameter(Phi)
        self.lsigma = nn.Parameter(lsigma)
        self.theta = theta
        self.f = f
        self.psi = psi
        self.kap = Phi.shape[1]
        self.optim_param_set = None
        if optim_param_set is None:
            optim_param_set = {'Lmb': True, 'G': False, 'Phi': False, 'lsigma': True}
        self.update_optim_flags(optim_param_set)

    def update_optim_flags(self, optim_param_set):
        self.optim_param_set = optim_param_set
        self.Lmb.requires_grad = optim_param_set['Lmb']
        self.G.requires_grad = optim_param_set['G']
        self.Phi.requires_grad = optim_param_set['Phi']
        self.lsigma.requires_grad = optim_param_set['lsigma']

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
        lsigma = self.lsigma

        f = self.f
        psi = self.psi
        Phi = self.Phi
        return negloglik_mvlatent(Lmb=Lmb, G=G, lsigma=lsigma, theta=theta, f=f, psi=psi, Phi=Phi)

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


def negloglik_mvlatent(Lmb, G, lsigma, theta, f, psi, Phi):
    kap = Phi.shape[1]
    m, n = f.shape

    def predmean_gp_(Vh, lmb, theta, thetanew, g):
        Rinv_g = Vh @ Vh.T @ g
        R_no = covmat(thetanew, theta, lmb)
        return R_no @ Rinv_g

    nll_gp = torch.zeros(kap)
    Gpred = torch.zeros_like(G)
    for k in range(kap):
        nll_gp[k], Vh = negloglik_gp(lmb=Lmb[k], theta=theta, g=G[:, k])
        Gpred[:, k] = predmean_gp_(Vh, Lmb[k], theta, theta, G[:, k])

    D = f - (psi + Phi @ Gpred.T)
    nll_diff = n * m * lsigma + 1 / 2 * torch.exp(-2 * lsigma) * (D.T @ D).sum()
    nll = (nll_diff + nll_gp.sum()).squeeze()
    return nll
