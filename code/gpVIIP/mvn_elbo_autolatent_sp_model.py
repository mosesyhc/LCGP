import torch
import torch.nn as nn
import torch.jit as jit
from matern_covmat import cormat, cov_sp
from likelihood import negloglik_gp_sp
from prediction import pred_gp_sp
from hyperparameter_tuning import parameter_clamping


class MVN_elbo_autolatent_sp(jit.ScriptModule):
    def __init__(self, F, theta,
                 p=None, thetai=None,
                 Phi=None, kap=None, pcthreshold=0.99,
                 lLmb=None, lsigma2=None,
                 initlLmb=True, initlsigma2=True,
                 choice_thetai='LHS', clamping=True):
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
        self.clamping = clamping
        if p is None and thetai is None:
            raise ValueError('Specify either p, (number of inducing points),'
                             ' or thetai, (inducing points).')
        elif p is None:
            self.p = thetai.shape[0]
            self.thetai = thetai
        else:
            self.p = p
            self.ip_choice(p=p, theta=theta, choice_thetai=choice_thetai)

        self.m, self.n = F.shape

        self.F = F

        self.standardize_F()

        if Phi is None:
            self.G, self.Phi, self.pcw, self.kap = self.__PCs(F=self.F, kap=kap, threshold=pcthreshold)

            Fhat = self.tx_F((self.Phi * self.pcw) @ self.G)
            Phi_mse = ((Fhat - self.Fraw) ** 2).mean()
            print('#PCs: {:d}, recovery mse: {:.3E}'.format(self.kap, Phi_mse.item()))
        else:
            self.G = Phi.T @ self.F  # (Phi.T / 1.) @ self.F
            self.Phi = Phi
            self.kap = kap
            self.pcw = torch.ones(kap)

        self.M = torch.zeros(self.kap, self.n)
        self.theta = theta
        self.d = theta.shape[1]
        if initlLmb or lLmb is None:
            llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                                torch.log(torch.std(theta, 0)))
            llmb = torch.cat((llmb, torch.Tensor([0])))
            lLmb = llmb.repeat(self.kap, 1)
            lLmb[:, -1] = torch.log(torch.var(self.G, 1))
        self.lLmb = nn.Parameter(lLmb)

        lnugGPs = torch.Tensor(-16 * torch.ones(self.kap))
        self.lnugGPs = nn.Parameter(lnugGPs)

        if initlsigma2 or lsigma2 is None:
            lsigma2 = torch.log(((self.Phi @ self.Phi.T @ self.F - self.F) ** 2).mean())
        self.lmse0 = lsigma2.item()
        self.lsigma2 = nn.Parameter(lsigma2)
        self.buildtime: float = 0.0

    # @jit.script_method
    def forward(self, theta0):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lnugGPs = self.lnugGPs

        if self.clamping:
            lLmb, lsigma2, lnugGPs = self.parameter_clamp(lLmb, lsigma2, lnugGPs)

        M = self.M
        theta = self.theta
        thetai = self.thetai

        kap = self.kap
        n0 = theta0.shape[0]

        ghat_sp = torch.zeros(kap, n0)
        for k in range(kap):
            ck = cormat(theta0, theta, llmb=lLmb[k])
            Delta_k_inv_diag, Qk_Rkinvh, logdet_Ck, \
                _, _ = cov_sp(theta=theta, thetai=thetai, llmb=lLmb[k])

            # C_sp_inv = torch.diag(Delta_inv_diag) - Q_Rinvh @ Q_Rinvh.T
            Ckinv_Mk = Delta_k_inv_diag * M[k] - Qk_Rkinvh @ (Qk_Rkinvh.T * M[k]).sum(1)

            ghat_sp[k] = ck @ Ckinv_Mk

        fhat = (self.Phi * self.pcw) @ ghat_sp
        fhat = self.tx_F(fhat)
        return fhat

    def negelbo(self):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lnugGPs = self.lnugGPs

        if self.clamping:
            lLmb, lsigma2, lnugGPs = self.parameter_clamp(lLmb, lsigma2, lnugGPs)

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
            negloggp_sp_k = negloglik_gp_sp(llmb=lLmb[k], theta=theta,
                                            thetai=thetai, g=M[k])
            negelbo += negloggp_sp_k


        residF = F - (self.Phi * self.pcw) @ M
        negelbo += m * n / 2 * lsigma2
        negelbo += 1 / (2 * lsigma2.exp()) * (residF ** 2).sum()
        negelbo -= 1 / 2 * torch.log(V).sum()

        negelbo += 8 * (lsigma2 - self.lmse0) ** 2

        return negelbo

    def compute_MV(self):
        lsigma2 = self.lsigma2
        lLmb = self.lLmb
        lnugGPs = self.lnugGPs

        if self.clamping:
            lLmb, lsigma2, lnugGPs = self.parameter_clamp(lLmb, lsigma2, lnugGPs)

        kap = self.kap
        theta = self.theta
        thetai = self.thetai

        G = self.G
        sigma2 = torch.exp(lsigma2)

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)
        for k in range(kap):
            Delta_k_inv_diag, Qk_Rkinvh, \
                logdet_Ck, ck_full_i, Ck_i = cov_sp(theta, thetai, lLmb[k])
            Dinv_k_diag = 1 / (1 + sigma2 * Delta_k_inv_diag)

            # F_Phik = ((Phi * pcw)[:, k] * F.T).sum(1)  #checked
            Tk = Ck_i + ck_full_i.T * (1/Delta_k_inv_diag + sigma2) @ ck_full_i  #checked

            W_Tk, U_Tk = torch.linalg.eigh(Tk)
            Tkinvh = U_Tk / W_Tk.abs().sqrt()

            Sk = ((1 - Dinv_k_diag) * ck_full_i.T).T
            Sk_Tkinvh = Sk @ Tkinvh

            Mk = Dinv_k_diag * G[k] + 1/sigma2 * Sk_Tkinvh @ (Sk_Tkinvh.T * G[k]).sum(1)
            M[k] = Mk
            # V[k] = 1 / (1/sigma2 + Delta_k_inv_diag - torch.diag(Qk_Rkinvh @ Qk_Rkinvh.T))  #
            V[k] = 1 / (1/sigma2 + Delta_k_inv_diag - (Qk_Rkinvh ** 2).sum(1))  # (Qk_Rkinvh ** 2).sum(0, 1)

        self.M = M
        self.V = V

    def predictmean(self, theta0):
        with torch.no_grad():
            self.compute_MV()
            fhat = self.forward(theta0)
        return fhat

    def predictcov(self, theta0):
        with torch.no_grad():
            self.compute_MV()
            theta = self.theta
            lLmb, lsigma2, lnugGPs = self.parameter_clamp(self.lLmb, self.lsigma2, self.lnugGPs)

            txPhi = (self.Phi * self.pcw * self.Fstd)
            V = self.V

            n0 = theta0.shape[0]
            kap = self.kap
            m = self.m

            predcov = torch.zeros(m, m, n0)

            predcov_g = torch.zeros(kap, n0)
            for k in range(kap):
                ck = cormat(theta0, theta, llmb=lLmb[k])
                Ck = cormat(theta, theta, llmb=lLmb[k])

                Wk, Uk = torch.linalg.eigh(Ck)
                Ukh = Uk / torch.sqrt(Wk)

                ck_Ckinvh = ck @ Ukh
                ck_Ckinv_Vkh = ck @ Ukh @ Ukh.T * torch.sqrt(V[k])

                predcov_g[k] = 1 - (ck_Ckinvh**2).sum(1) + (ck_Ckinv_Vkh**2).sum(1)

            for i in range(n0):
                predcov[:, :, i] = torch.exp(lsigma2) * torch.eye(m) + \
                                   txPhi * predcov_g[:, i] @ txPhi.T
        return predcov

    def predictvar(self, theta0):
        predcov = self.predictcov(theta0)

        n0 = theta0.shape[0]
        m = self.m
        predvar = torch.zeros(m, n0)
        for i in range(n0):
            predvar[:, i] = predcov[:, :, i].diag()
        return predvar

    def test_mse(self, theta0, f0):
        with torch.no_grad():
            m, n = f0.shape
            fhat = self.predictmean(theta0)
            return ((fhat - f0) ** 2).sum() / (m * n)

    def test_rmse(self, theta0, f0):
        return torch.sqrt(self.test_mse(theta0=theta0, f0=f0))

    def test_individual_error(self, theta0, f0):
        with torch.no_grad():
            fhat = self.predictmean(theta0)
            return torch.sqrt((fhat - f0) ** 2).mean(0)

    def standardize_F(self):
        if self.F is not None:
            F = self.F
            self.Fraw = F.clone()
            self.Fmean = F.mean(1).unsqueeze(1)
            self.Fstd = F.std(1).unsqueeze(1)
            self.F = (F - self.Fmean) / self.Fstd

    def tx_F(self, Fs):
        return Fs * self.Fstd + self.Fmean

    @staticmethod
    def parameter_clamp(lLmb, lsigma2, lnugs):
        # clamping
        lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5, 2.5)))).T
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, 0)))
        lnugs = parameter_clamping(lnugs, torch.tensor((-20, -8)))

        return lLmb, lsigma2, lnugs

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

    @staticmethod
    def __PCs(F, kap=None, threshold=0.99):
        m, n = F.shape

        Phi, S, _ = torch.linalg.svd(F, full_matrices=False)
        v = (S ** 2).cumsum(0) / (S ** 2).sum()

        if kap is None:
            kap = int(torch.argwhere(v > threshold)[0][0] + 1)

        assert Phi.shape[1] == m
        Phi = Phi[:, :kap]
        S = S[:kap]

        pcw = ((S ** 2).abs() / n).sqrt()

        G = (Phi / pcw).T @ F  # kap x n
        return G, Phi, pcw, kap
