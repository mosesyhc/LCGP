import torch
import torch.nn as nn
from matern_covmat import covmat
from likelihood import negloglik_gp
from prediction import pred_gp


class MVN_elbo_autoMuV(nn.Module):
    def __init__(self, Lmb, lsigma2, psi, Phi, F, theta, initLmb=True):
        super().__init__()
        self.kap = Phi.shape[1]
        if initLmb:
            lmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                                torch.log(torch.std(theta, 0)))
            lmb = torch.cat((lmb, torch.Tensor([0])))
            Lmb = lmb.repeat(self.kap, 1)
        self.Lmb = nn.Parameter(Lmb)
        self.lsigma2 = nn.Parameter(lsigma2)
        self.psi = psi
        self.Phi = Phi
        self.F = F
        self.theta = theta
        self.m, self.n = F.shape
        self.Mu = torch.zeros((Phi.shape[1], self.n))

    def forward(self, theta0):
        Lmb = self.Lmb
        Mu = self.Mu

        psi = self.psi
        Phi = self.Phi
        theta = self.theta

        kap = self.kap
        n0 = theta0.shape[0]

        ghat = torch.zeros(kap, n0)
        for k in range(kap):
            ghat[k], _ = pred_gp(llmb=Lmb[k], theta=theta, thetanew=theta0, g=Mu[k])

        fhat = psi + Phi @ ghat
        return fhat

    def negelbo(self):
        Lmb = self.Lmb
        Mu = self.Mu
        V = self.V
        lsigma2 = self.lsigma2

        psi = self.psi
        Phi = self.Phi
        F = self.F
        theta = self.theta

        m = self.m
        n = self.n
        kap = self.kap

        def predmean_gp_(Vh, lmb, theta, thetanew, g):
            Rinv_g = Vh @ Vh.T @ g
            R_no = covmat(thetanew, theta, lmb)
            return R_no @ Rinv_g

        negelbo = 1/2 * torch.exp(-lsigma2) * torch.sum(torch.diag(Phi.T @ Phi) @ V) + \
            m*n/2 * lsigma2
        if V.requires_grad:
            negelbo += -1/2 * torch.sum(torch.log(V))

        Mupred = torch.zeros((kap, n))
        for k in range(kap):
            negloggp_k, Vh_k = negloglik_gp(llmb=Lmb[k], x=theta, g=Mu[k])
            negelbo += negloggp_k
            negelbo += 1/2 * torch.diag(Vh_k @ Vh_k.T) @ V[k]
            Mupred[k] = predmean_gp_(Vh=Vh_k, lmb=Lmb[k], theta=theta,
                                     thetanew=theta, g=Mu[k])

        residF = F - (psi + Phi @ Mupred)

        negelbo += 1/2 * torch.exp(-lsigma2) * (residF.T @ residF).sum()
        # print(lsigma2, V.mean())
        return negelbo.squeeze()


    def test_mse(self, theta0, f0):
        fhat = self.forward(theta0)
        return ((fhat - f0) ** 2).mean()
