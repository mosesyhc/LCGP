import torch
import torch.nn as nn
from matern_covmat import covmat, cov_sp
from likelihood import negloglik_gp, negloglik_gp_sp
from prediction import pred_gp, pred_gp_sp


class MVN_elbo_autolatent_sp(nn.Module):
    def __init__(self, Lmb, lsigma2, psi, Phi, F, theta, thetai, initLmb=True):
        super().__init__()
        if psi.ndim < 2:
            psi = psi.unsqueeze(1)
        self.kap = Phi.shape[1]
        self.m, self.n = F.shape
        self.p = thetai.shape[0]
        if initLmb:
            lmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                                torch.log(torch.std(theta, 0)))
            lmb = torch.cat((lmb, torch.Tensor([0])))
            Lmb = lmb.repeat(self.kap, 1)
            Lmb[:, -1] = torch.log(torch.var(Phi.T @ (F - psi), 1))
        self.Lmb = nn.Parameter(Lmb)
        self.lsigma2 = nn.Parameter(lsigma2)
        self.M = torch.zeros(self.kap, self.n)
        self.V = torch.zeros(self.kap, self.n)
        self.psi = psi
        self.Phi = Phi
        self.F = F
        self.theta = theta
        self.thetai = thetai

    def forward(self, theta0):
        Lmb = self.Lmb
        M = self.M
        lsigma2 = self.lsigma2

        psi = self.psi
        Phi = self.Phi
        theta = self.theta
        thetai = self.thetai

        kap = self.kap
        n0 = theta0.shape[0]

        ghat = torch.zeros(kap, n0)
        ghat_sp = torch.zeros_like(ghat)
        for k in range(kap):
            ghat_sp[k], _ = pred_gp_sp(lmb=Lmb[k], theta=theta, thetai=thetai, thetanew=theta0, lsigma2=lsigma2, g=M[k])

        fhat = psi + Phi @ ghat_sp
        return fhat

    def negelbo(self):
        Lmb = self.Lmb
        M = self.M
        V = self.V
        lsigma2 = self.lsigma2
        sigma2 = torch.exp(lsigma2)

        psi = self.psi
        Phi = self.Phi
        F = self.F
        theta = self.theta
        thetai = self.thetai

        m = self.m
        n = self.n
        kap = self.kap
        p = self.p

        negelbo = torch.zeros(1)
        for k in range(kap):
            Lmb_inv_diag, Qk_half, logdet_Ck = cov_sp(theta, thetai, lsigma2, Lmb[k])

            Dinv_k_diag = 1 / (sigma2 * Lmb_inv_diag + 1)
            Sk = torch.eye(p) - sigma2 * (Qk_half.T * Dinv_k_diag) @ Qk_half
            W_Sk, U_Sk = torch.linalg.eigh(Sk)
            Tk_half = (Dinv_k_diag * Qk_half.T).T @ U_Sk / torch.sqrt(torch.abs(W_Sk)) @ U_Sk.T

            V[k] = 1 / (1 / sigma2 + Lmb_inv_diag - torch.diag(Qk_half @ Qk_half.T))  #
            M[k] = Dinv_k_diag * (Phi[:, k] * (F - psi).T).sum(1) + sigma2 * Tk_half @ Tk_half.T @ (Phi[:, k] * (F - psi).T).sum(1)

            negloggp_sp_k = negloglik_gp_sp(lmb=Lmb[k], theta=theta, thetai=thetai, lsigma2=lsigma2, g=M[k].clone())
            negelbo += negloggp_sp_k

        residF = F - (psi + Phi @ M)
        negelbo += m*n/2 * lsigma2
        negelbo += 1/2 * torch.exp(-lsigma2) * (residF.T @ residF).sum()
        negelbo -= 1/2 * torch.log(V).sum()

        self.M = M
        self.V = V
        return negelbo.squeeze()

    def test_mse(self, theta0, f0):
        fhat = self.forward(theta0)
        return ((fhat - f0) ** 2).mean()
