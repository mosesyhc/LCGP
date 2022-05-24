import torch
import torch.jit as jit
import torch.nn as nn
from matern_covmat import cov_sp
from likelihood import negloglik_gp_sp
from prediction import pred_gp_sp
from hyperparameter_tuning import parameter_clamping


class MVN_elbo_autolatent_sp(jit.ScriptModule):
    def __init__(self, Phi, F, theta, p=None, thetai=None,
                 lLmb=None, lsigma2=None, initlLmb=True, initlsigma2=True,
                 choice_thetai='LHS'):
        """

        :param Phi:
        :param F:
        :param theta:
        :param p:
        :param thetai:
        :param lLmb:
        :param lsigma2:
        :param initlLmb:
        :param initlsigma2:
        :param choice_thetai:
        """
        super().__init__()
        self.method = 'MVIP'
        if p is None and thetai is None:
            raise ValueError('Specify either p, (number of inducing points),'
                             ' or thetai, (inducing points).')
        elif p is None:
            self.p = thetai.shape[0]
            self.thetai = thetai
        else:
            self.p = p
            self.ip_choice(p=p, theta=theta, choice_thetai=choice_thetai)

        self.kap = Phi.shape[1]
        self.m, self.n = F.shape
        self.M = torch.zeros(self.kap, self.n)
        self.V = torch.zeros(self.kap, self.n)
        self.Phi = Phi

        self.F = F
        self.Fraw = None
        self.Fstd = None
        self.Fmean = None
        self.standardize_F()

        self.theta = theta
        self.d = theta.shape[1]
        if initlLmb or lLmb is None:
            llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                               torch.log(torch.std(theta, 0)))
            llmb = torch.cat((llmb, torch.Tensor([0])))
            lLmb = llmb.repeat(self.kap, 1)
            lLmb[:, -1] = torch.log(torch.var(Phi.T @ self.F, 1))
        self.lLmb = nn.Parameter(lLmb)
        if initlsigma2 or lsigma2 is None:
            lsigma2 = torch.log(((Phi @ Phi.T @ self.F - self.F)**2).mean())
        self.lsigma2 = nn.Parameter(lsigma2)  # nn.Parameter(torch.tensor((-8,)), requires_grad=False)
        self.buildtime:float = 0.0

    @jit.script_method
    def forward(self, theta0):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lLmb, lsigma2 = self.parameter_clamp(lLmb, lsigma2)

        M = self.M
        Phi = self.Phi
        theta = self.theta
        thetai = self.thetai

        kap = self.kap
        n0 = theta0.shape[0]

        ghat_sp = torch.zeros(kap, n0)
        for k in range(kap):
            ghat_sp[k], _ = pred_gp_sp(llmb=lLmb[k], theta=theta, thetanew=theta0, thetai=thetai, g=M[k])

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

        M = self.M
        V = self.V
        negelbo = 0
        for k in range(kap):
            negloggp_sp_k = negloglik_gp_sp(llmb=lLmb[k], theta=theta, thetai=thetai, g=M[k])
            negelbo += negloggp_sp_k

        residF = F - (Phi @ M)
        negelbo += m*n/2 * lsigma2
        negelbo += 1/(2 * sigma2) * (residF ** 2).sum()
        negelbo -= 1/2 * torch.log(V).sum()

        return negelbo

    def compute_MV(self):
        lsigma2 = self.lsigma2
        lLmb = self.lLmb
        lLmb, lsigma2 = self.parameter_clamp(lLmb, lsigma2)

        kap = self.kap
        theta = self.theta
        thetai = self.thetai

        Phi = self.Phi
        F = self.F
        sigma2 = torch.exp(lsigma2)

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)
        for k in range(kap):
            Delta_k_inv_diag, Qk_Rkinvh, logdet_Ck, ck_full_i, Ck_i = cov_sp(theta, thetai, lLmb[k])
            Dinv_k_diag = 1 / (sigma2 * Delta_k_inv_diag + 1)

            F_Phik = (Phi[:, k] * F.T).sum(1)
            Tk = Ck_i + ck_full_i.T * ((1 - Dinv_k_diag) / sigma2) @ ck_full_i
            W_Tk, U_Tk = torch.linalg.eigh(Tk)

            Sk_Tkinvh = ((1 - Dinv_k_diag) * ck_full_i.T).T @ (U_Tk * torch.sqrt(1 / W_Tk.abs())) @ U_Tk.T
            Mk = Dinv_k_diag * F_Phik + 1/sigma2 * Sk_Tkinvh @ Sk_Tkinvh.T @ F_Phik
            M[k] = Mk
            # V[k] = 1 / (1/sigma2 + Delta_k_inv_diag - torch.diag(Qk_Rkinvh @ Qk_Rkinvh.T))  #
            V[k] = 1 / (1/sigma2 + Delta_k_inv_diag - (Qk_Rkinvh ** 2).sum(1))  # (Qk_Rkinvh ** 2).sum(0, 1)

        self.M = M.detach_()
        self.V = V.detach_()

    def predict(self, theta0):
        self.compute_MV()
        fhat = self.forward(theta0)
        return(fhat)

    def test_mse(self, theta0, f0):
        with torch.no_grad():
            fhat = self.predict(theta0)
            return ((fhat - f0) ** 2).mean()

    def test_rmse(self, theta0, f0):
        return torch.sqrt(self.test_mse(theta0=theta0, f0=f0))

    def test_individual_error(self, theta0, f0):
        with torch.no_grad():
            fhat = self.predict(theta0)
            return torch.sqrt((fhat - f0)**2).mean(0)

    def standardize_F(self):
        if self.F is not None:
            F = self.F
            self.Fraw = F.clone()
            self.Fmean = F.mean(1).unsqueeze(1)
            self.Fstd = F.std(1).unsqueeze(1)
            self.F = (F - self.Fmean) / self.Fstd

    @staticmethod
    def parameter_clamp(lLmb, lsigma2):
        # clamping
        lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5, 2.5)))).T
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, -1)))

        return lLmb, lsigma2

    def ip_choice(self, p, theta, choice_thetai):
        if choice_thetai == 'LHS':  # assume [0, 1]^d
            from scipy.stats.qmc import LatinHypercube
            sampler = LatinHypercube(d=theta.shape[1])
            self.thetai = torch.tensor(sampler.random(p))
        elif choice_thetai == 'Sobol':
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=theta.shape[1])
            self.thetai = torch.tensor(sampler.random(p))
        elif choice_thetai == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans_gen = KMeans(p).fit(theta)
            self.thetai = torch.tensor(kmeans_gen.cluster_centers_)
        else:
            raise ValueError('Currently only \'LHS\', \'Sobol\', '
                             '\'kmeans\' are supported for choosing inducing points.')
