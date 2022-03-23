import torch
import torch.nn as nn
from matern_covmat import covmat, cov_sp
from likelihood import negloglik_gp, negloglik_gp_sp
from prediction import pred_gp, pred_gp_sp


class MVN_elbo_autolatent_sp(nn.Module):
    def __init__(self, Lmb, lsigma2, psi, Phi, F, theta, thetai, initLmb=True):
        super().__init__()
        self.kap = Phi.shape[1]
        self.m, self.n = F.shape
        if initLmb:
            lmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                                torch.log(torch.std(theta, 0)))
            lmb = torch.cat((lmb, torch.Tensor([0])))
            Lmb = lmb.repeat(self.kap, 1)
            Lmb[:, -1] = torch.log(torch.var(Phi.T @ (F - psi), 1))
        self.Lmb = nn.Parameter(Lmb)
        self.lsigma2 = lsigma2
        self.Mu = torch.zeros(self.kap, self.n)
        self.V = torch.zeros(self.kap, self.n)
        self.psi = psi
        self.Phi = Phi
        self.F = F
        self.theta = theta
        self.thetai = thetai

    def forward(self, theta0):
        Lmb = self.Lmb
        Mu = self.Mu

        psi = self.psi
        Phi = self.Phi
        theta = self.theta
        thetai = self.thetai

        kap = self.kap
        n0 = theta0.shape[0]

        ghat = torch.zeros(kap, n0)
        ghat_sp = torch.zeros_like(ghat)
        for k in range(kap):
            ghat[k], _ = pred_gp(lmb=Lmb[k], theta=theta, thetanew=theta0, g=Mu[k])
            ghat_sp[k], _ = pred_gp_sp(lmb=Lmb[k], theta=theta, thetai=thetai, thetanew=theta0, g=Mu[k])
            print('implementation diff in pred:', ((ghat[k] - ghat_sp[k])**2).sum())

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
        thetai = self.thetai

        m = self.m
        n = self.n
        kap = self.kap

        def predmean_gp_(Vh, lmb, theta, thetanew, g):
            Rinv_g = Vh @ Vh.T @ g
            R_no = covmat(thetanew, theta, lmb)
            return R_no @ Rinv_g

        Vinv = torch.zeros_like(V)
        negelbo = torch.zeros(1)
        for k in range(kap):
            C_k, C_k_inv, logdet_C_k = cov_sp(theta, thetai, Lmb[k])
            # W_k, U_k = torch.linalg.eigh(C_k)
            # Winv_k = 1 / W_k
            V[k] = 1 / torch.exp(-lsigma2) + torch.diag(C_k_inv)
            Mu[k] = torch.linalg.solve(torch.eye(n) + torch.exp(lsigma2) * C_k_inv, Phi[:, k] @ (F - psi))

            negloggp_k, _ = negloglik_gp(lmb=Lmb[k], theta=theta, g=Mu[k].clone())
            negloggp_sp_k = negloglik_gp_sp(lmb=Lmb[k], theta=theta, thetai=thetai, g=Mu[k].clone())
            print('implementation diff in neglog:', (negloggp_k - negloggp_sp_k)**2)
            negelbo += negloggp_k



        residF = F - (psi + Phi @ Mu)
        negelbo += m*n/2 * lsigma2
        negelbo += 1/2 * torch.exp(-lsigma2) * (residF.T @ residF).sum()
        negelbo -= 1/2 * torch.log(V).sum()

        self.Mu = Mu
        self.V = V
        return negelbo.squeeze()


    def test_mse(self, theta0, f0):
        fhat = self.forward(theta0)
        return ((fhat - f0) ** 2).mean()
