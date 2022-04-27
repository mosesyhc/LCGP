import torch
import torch.nn as nn
from matern_covmat import covmat
from likelihood import negloglik_gp
from prediction import pred_gp
from hyperparameter_tuning import parameter_clamping, C_LLMB, C_LSIGMA2

class MVN_elbo_autolatent(nn.Module):
    def __init__(self, lLmb, lsigma2, Phi, F, theta, initlLmb=True, initlsigma2=True):
        super().__init__()
        self.kap = Phi.shape[1]
        self.m, self.n = F.shape
        self.M = torch.zeros(self.kap, self.n)
        self.Phi = Phi

        self.F = F
        self.Fraw = None
        self.Fstd = None
        self.Fmean = None
        self.standardize_F()

        self.theta = theta
        if initlLmb:
            llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                               torch.log(torch.std(theta, 0)))
            llmb = torch.cat((llmb, torch.Tensor([0])))
            lLmb = llmb.repeat(self.kap, 1)
            lLmb[:, -1] = torch.log(torch.var(Phi.T @ self.F, 1))
        self.lLmb = nn.Parameter(lLmb)
        if initlsigma2:
            lsigma2 = torch.log(((Phi @ Phi.T @ self.F - self.F)**2).mean())
        self.lsigma2 = nn.Parameter(lsigma2)  # nn.Parameter(torch.tensor((-8,)), requires_grad=False)

    def forward(self, theta0):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lLmb, lsigma2 = self.parameter_clamp(lLmb, lsigma2)

        M = self.M
        Phi = self.Phi
        theta = self.theta

        kap = self.kap
        n0 = theta0.shape[0]

        ghat = torch.zeros(kap, n0)
        for k in range(kap):
            ghat[k], _ = pred_gp(llmb=lLmb[k], theta=theta, thetanew=theta0, lsigma2=lsigma2, g=M[k])

        fhat = Phi @ ghat
        fhat = (fhat * self.Fstd) + self.Fmean
        return fhat

    def negelbo(self):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lLmb, lsigma2 = self.parameter_clamp(lLmb, lsigma2)

        Phi = self.Phi
        F = self.F
        theta = self.theta

        m = self.m
        n = self.n
        kap = self.kap

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)

        negelbo = torch.zeros(1)
        for k in range(kap):
            C_k = covmat(theta, theta, lLmb[k])
            W_k, U_k = torch.linalg.eigh(C_k)
            Winv_k = 1 / W_k
            Mk = torch.linalg.solve(torch.eye(n) + torch.exp(lsigma2) * U_k @ torch.diag(Winv_k) @ U_k.T, Phi[:, k] @ F)

            negloggp_k, _ = negloglik_gp(llmb=lLmb[k], theta=theta, g=Mk)
            negelbo += negloggp_k

            M[k] = Mk
            V[k] = 1 / (torch.exp(-lsigma2) + torch.diag(U_k @ torch.diag(Winv_k) @ U_k.T))

        residF = F - (Phi @ M)
        negelbo += m*n/2 * lsigma2
        negelbo += 1/2 * torch.exp(-lsigma2) * (residF.T @ residF).sum()
        negelbo -= 1/2 * torch.log(V).sum()

        return negelbo.squeeze()

    def create_MV(self):
        kap = self.kap
        theta = self.theta
        lsigma2 = self.lsigma2
        lLmb = self.lLmb

        n = self.n
        Phi = self.Phi
        F = self.F

        for k in range(kap):
            C_k = covmat(theta, theta, lLmb[k])
            W_k, U_k = torch.linalg.eigh(C_k)
            Winv_k = 1 / W_k
            Mk = torch.linalg.solve(torch.eye(n) + torch.exp(lsigma2) * U_k @ torch.diag(Winv_k) @ U_k.T, Phi[:, k] @ F)
            self.M[k] = Mk

    def test_mse(self, theta0, f0):
        fhat = self.forward(theta0)
        return ((fhat - f0) ** 2).mean()

    def test_rmse(self, theta0, f0):
        return torch.sqrt(self.test_mse(theta0=theta0, f0=f0))

    def parameter_clamp(self, lLmb, lsigma2):
        # clamping
        lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5, 2.5)), c=C_LLMB)).T
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, -1)), c=C_LSIGMA2)
        return lLmb, lsigma2

    def standardize_F(self):
        if self.F is not None:
            F = self.F
            self.Fraw = F.clone()
            self.Fmean = F.mean(1).unsqueeze(1)
            self.Fstd = F.std(1).unsqueeze(1)
            self.F = (F - self.Fmean) / self.Fstd