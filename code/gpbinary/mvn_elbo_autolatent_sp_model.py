import torch
import torch.nn as nn
from matern_covmat import cov_sp
from likelihood import negloglik_gp_sp
from prediction import pred_gp_sp
from hyperparameter_tuning import parameter_clamping


class MVN_elbo_autolatent_sp(nn.Module):
    def __init__(self, Phi, F, theta, p=None, thetai=None,
                 lLmb=None, lsigma2=None, initlLmb=True, initlsigma2=True,
                 init_thetai=False, choice_thetai='LHS'):  #psi
        super().__init__()
        if p is None and thetai is None:
            raise ValueError('Specify either p, (number of inducing points),'
                             ' or thetai, (inducing points).')
        if p is None:
            self.p = thetai.shape[0]
            self.thetai = thetai
        else:
            self.p = p
            if choice_thetai == 'LHS':  # assume [0, 1]^d
                from scipy.stats.qmc import LatinHypercube
                sampler = LatinHypercube(d=theta.shape[1])
                self.thetai = sampler.random(p)
            elif choice_thetai == 'kmeans':
                from sklearn.cluster import KMeans
                kmeans_gen = KMeans(p).fit(theta)
                self.thetai = kmeans_gen.cluster_centers_
            else:
                raise ValueError('Currently only LHS or kmeans is supported for choosing inducing points')

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
        self.d = theta.shape[1]
        if initlLmb:
            llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                               torch.log(torch.std(theta, 0)))
            llmb = torch.cat((llmb, torch.Tensor([0])))
            lLmb = llmb.repeat(self.kap, 1)
            lLmb[:, -1] = torch.log(torch.var(Phi.T @ self.F, 1))
        self.lLmb = nn.Parameter(lLmb)
        if initlsigma2:
            lsigma2 = nn.Parameter(torch.log(((Phi @ Phi.T @ self.F - self.F)**2).mean()))
        self.lsigma2 = nn.Parameter(lsigma2)  # nn.Parameter(torch.tensor((-8,)), requires_grad=False)

    def forward(self, theta0):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lLmb, lsigma2 = self.parameter_clamp(lLmb, lsigma2)

        Phi = self.Phi
        theta = self.theta
        thetai = self.thetai

        kap = self.kap
        n0 = theta0.shape[0]

        M = self.M
        ghat_sp = torch.zeros(kap, n0)
        for k in range(kap):
            ghat_sp[k], _ = pred_gp_sp(llmb=lLmb[k], theta=theta, thetai=thetai, thetanew=theta0, lsigma2=lsigma2, g=M[k])
        fhat = Phi @ ghat_sp
        fhat = (fhat * self.Fstd) + self.Fmean
        return fhat

    def negelbo(self):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lLmb, lsigma2 = self.parameter_clamp(lLmb, lsigma2)

        sigma2 = torch.exp(lsigma2)
        Phi = self.Phi
        F = self.F
        theta = self.theta
        thetai = self.thetai

        m = self.m
        n = self.n
        kap = self.kap
        p = self.p

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)

        negelbo = torch.zeros(1)
        for k in range(kap):
            Delta_inv_diag, Qk_half, logdet_Ck = cov_sp(theta, thetai, lsigma2, lLmb[k])

            Dinv_k_diag = 1 / (sigma2 * Delta_inv_diag + 1)
            Sk = torch.eye(p) - sigma2 * (Qk_half.T * Dinv_k_diag) @ Qk_half
            W_Sk, U_Sk = torch.linalg.eigh(Sk)
            Tk_half = (Dinv_k_diag * Qk_half.T).T @ U_Sk / torch.sqrt(W_Sk) @ U_Sk.T

            Mk = Dinv_k_diag * (Phi[:, k] * F.T).sum(1) + sigma2 * Tk_half @ Tk_half.T @ (Phi[:, k] * F.T).sum(1)

            negloggp_sp_k = negloglik_gp_sp(llmb=lLmb[k], theta=theta, thetai=thetai, lsigma2=lsigma2, g=Mk,
                                            Delta_inv_diag=Delta_inv_diag, Q_half=Qk_half, logdet_C=logdet_Ck)
            negelbo += negloggp_sp_k
            M[k] = Mk
            V[k] = 1 / (1 / sigma2 + Delta_inv_diag - torch.diag(Qk_half @ Qk_half.T))  #

        residF = F - (Phi @ M)
        negelbo += m*n/2 * lsigma2
        negelbo += 1/2 * torch.exp(-lsigma2) * (residF.T @ residF).sum()
        negelbo -= 1/2 * torch.log(V).sum()

        return negelbo.squeeze()

    def create_MV(self):
        lsigma2 = self.lsigma2
        lLmb = self.lLmb
        lLmb, lsigma2 = self.parameter_clamp(lLmb, lsigma2)

        kap = self.kap
        theta = self.theta
        thetai = self.thetai

        p = self.p
        Phi = self.Phi
        F = self.F
        sigma2 = torch.exp(lsigma2)

        for k in range(kap):
            Delta_inv_diag, Qk_half, logdet_Ck = cov_sp(theta, thetai, lsigma2, lLmb[k])

            Dinv_k_diag = 1 / (sigma2 * Delta_inv_diag + 1)
            Sk = torch.eye(p) - sigma2 * (Qk_half.T * Dinv_k_diag) @ Qk_half
            W_Sk, U_Sk = torch.linalg.eigh(Sk)
            Tk_half = (Dinv_k_diag * Qk_half.T).T @ U_Sk / torch.sqrt(W_Sk) @ U_Sk.T

            Mk = Dinv_k_diag * (Phi[:, k] * F.T).sum(1) + sigma2 * Tk_half @ Tk_half.T @ (Phi[:, k] * F.T).sum(1)
            self.M[k] = Mk

    def test_mse(self, theta0, f0):
        with torch.no_grad():
            fhat = self.forward(theta0)
            return ((fhat - f0) ** 2).mean()

    def test_rmse(self, theta0, f0):
        return torch.sqrt(self.test_mse(theta0=theta0, f0=f0))

    def test_individual_error(self, theta0, f0):
        with torch.no_grad():
            fhat = self.forward(theta0)
            return torch.sqrt((fhat - f0)**2).mean(0)

    def standardize_F(self):
        if self.F is not None:
            F = self.F
            self.Fraw = F.clone()
            self.Fmean = F.mean(1).unsqueeze(1)
            self.Fstd = F.std(1).unsqueeze(1)
            self.F = (F - self.Fmean) / self.Fstd

    def parameter_clamp(self, lLmb, lsigma2):
        # clamping
        lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5, 2.5)))).T
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, -1)))

        return lLmb, lsigma2
