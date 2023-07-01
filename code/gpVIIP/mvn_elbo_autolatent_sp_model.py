import torch
import torch.nn as nn
import torch.jit as jit
from matern_covmat import covmat, cov_sp
from likelihood import negloglik_gp_sp
from hyperparameter_tuning import parameter_clamping

JIT = False
if JIT:
    Module = jit.ScriptModule
else:
    Module = nn.Module


class MVN_elbo_autolatent_sp(Module):
    def __init__(self, F, theta,
                 p=None, thetai=None,
                 Phi=None, kap=None, pcthreshold=0.9999,
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
        self.V = torch.zeros(self.kap, self.n)
        self.Delta_inv_diags = torch.zeros((self.kap, self.n))
        self.QRinvhs = torch.zeros((self.kap, self.n, self.p))
        self.Ciinvhs = torch.zeros((self.kap, self.p, self.p))
        self.Rinvhs = torch.zeros((self.kap, self.p, self.p))

        self.theta = theta
        self.d = theta.shape[1]
        if initlLmb or lLmb is None:
            llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                                torch.log(torch.std(theta, 0)))
            llmb = torch.cat((llmb, torch.Tensor([0])))
            lLmb = llmb.repeat(self.kap, 1)
            lLmb[:, -1] = torch.log(torch.var(self.G, 1))
        self.lLmb = nn.Parameter(lLmb)

        self.lnugGPs = nn.Parameter(torch.Tensor(-14 * torch.ones(self.kap)))
        self.ltau2GPs = nn.Parameter(torch.Tensor(torch.zeros(self.kap)))

        if initlsigma2 or lsigma2 is None:
            Fhat = (self.Phi * self.pcw) @ self.G
            lsigma2 = torch.max(torch.log(((Fhat - self.F) ** 2).mean()), torch.log(0.1 * (self.F**2).mean()))
        self.lmse0 = lsigma2.item()
        self.lsigma2 = nn.Parameter(lsigma2)
        self.buildtime: float = 0.0

    # @jit.script_method
    
    def forward(self, theta0):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lnugGPs = self.lnugGPs
        ltau2GPs = self.ltau2GPs

        if self.clamping:
            lLmb, lsigma2, lnugGPs, ltau2GPs = self.parameter_clamp(lLmb, lsigma2, lnugGPs, ltau2GPs)

        M = self.M
        QRinvhs = self.QRinvhs
        Rinvhs = self.Rinvhs
        thetai = self.thetai

        kap = self.kap

        n0 = theta0.shape[0]
        ghat_sp = torch.zeros(kap, n0)
        for k in range(kap):
            cki = covmat(theta0, thetai, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k])
            Ckinv_Mk = Rinvhs[k] @ (QRinvhs[k].T * M[k]).sum(1)

            ghat_sp[k] = cki @ Ckinv_Mk

        fhat = (self.Phi * self.pcw) @ ghat_sp
        fhat = self.tx_F(fhat)
        return fhat

    
    def negelbo(self):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lnugGPs = self.lnugGPs
        ltau2GPs = self.ltau2GPs

        if self.clamping:
            lLmb, lsigma2, lnugGPs, ltau2GPs = self.parameter_clamp(lLmb, lsigma2, lnugGPs, ltau2GPs)

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
            negloggp_sp_k, Cinvkdiag_sp = negloglik_gp_sp(llmb=lLmb[k], lnug=lnugGPs[k], ltau2=ltau2GPs[k],
                                            theta=theta, thetai=thetai, g=M[k])
            negelbo += negloggp_sp_k
            negelbo += 1 / 2 * (Cinvkdiag_sp * V[k]).sum()

        residF = F - (self.Phi * self.pcw) @ M
        negelbo += m * n / 2 * lsigma2
        negelbo += 1 / (2 * lsigma2.exp()) * (residF ** 2).sum()
        negelbo -= 1 / 2 * torch.log(V).sum()
        negelbo += 1 / (2 * lsigma2.exp()) * V.sum()

        negelbo += 2 * (lsigma2 - self.lmse0) ** 2

        return negelbo

    @torch.no_grad()
    def compute_MV(self):
        lsigma2 = self.lsigma2
        lLmb = self.lLmb
        lnugGPs = self.lnugGPs
        ltau2GPs = self.ltau2GPs

        if self.clamping:
            lLmb, lsigma2, lnugGPs, ltau2GPs = self.parameter_clamp(lLmb, lsigma2, lnugGPs, ltau2GPs)

        kap = self.kap
        theta = self.theta
        thetai = self.thetai

        G = self.G
        sigma2 = torch.exp(lsigma2)

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)

        Delta_inv_diags = torch.zeros((self.kap, self.n))
        QRinvhs = torch.zeros((self.kap, self.n, self.p))
        Rinvhs = torch.zeros((self.kap, self.p, self.p))
        Ciinvhs = torch.zeros((self.kap, self.p, self.p))

        for k in range(kap):
            Delta_k_inv_diag, Qk, Rkinvh, Qk_Rkinvh, \
                logdet_Ck, ck_full_i, Ck_i = cov_sp(theta=theta, thetai=thetai, llmb=lLmb[k], llmb0=ltau2GPs[k],
                                                    lnug=lnugGPs[k])

            W_Cki, U_Cki = torch.linalg.eigh(Ck_i)
            Ckiinvh = U_Cki / W_Cki.abs().sqrt()
            Ciinvhs[k] = Ckiinvh

            Dinvk_diag = 1 / (1 + sigma2 * Delta_k_inv_diag)
            D2invk_diag = 1 / (1 / Delta_k_inv_diag + sigma2)

            Tk = Ck_i + ck_full_i.T * D2invk_diag @ ck_full_i  #checked

            W_Tk, U_Tk = torch.linalg.eigh(Tk)
            Tkinvh = U_Tk / W_Tk.abs().sqrt()

            Sk = (D2invk_diag * ck_full_i.T).T
            Sk_Tkinvh = Sk @ Tkinvh

            Mk = Dinvk_diag * G[k] + sigma2 * Sk_Tkinvh @ (Sk_Tkinvh.T * G[k]).sum(1)

            M[k] = Mk
            V[k] = 1 / (1 / sigma2 + (Delta_k_inv_diag - (Qk_Rkinvh ** 2).sum(1)))

            Delta_inv_diags[k] = Delta_k_inv_diag
            QRinvhs[k] = Qk_Rkinvh
            Rinvhs[k] = Rkinvh

        self.M = M
        self.V = V
        self.Delta_inv_diags = Delta_inv_diags
        self.QRinvhs = QRinvhs
        self.Ciinvhs = Ciinvhs
        self.Rinvhs = Rinvhs

    
    def predictmean(self, theta0):
        with torch.no_grad():
            self.compute_MV()
            fhat = self.forward(theta0)
        return fhat

    
    def predictcov(self, theta0):
        with torch.no_grad():
            self.compute_MV()
            theta = self.theta
            thetai = self.thetai
            lLmb, lsigma2, lnugGPs, ltau2GPs = self.parameter_clamp(self.lLmb, self.lsigma2, self.lnugGPs, self.ltau2GPs)

            txPhi = (self.Phi * self.pcw * self.Fstd)
            V = self.V

            n0 = theta0.shape[0]
            kap = self.kap
            m = self.m

            Delta_inv_diags = self.Delta_inv_diags
            QRinvhs = self.QRinvhs
            Ciinvhs = self.Ciinvhs
            Rinvhs = self.Rinvhs

            term1 = torch.zeros(kap, n0)
            term2 = torch.zeros(kap, n0)

            predcov = torch.zeros(m, m, n0)
            predcov_g = torch.zeros(kap, n0)
            for k in range(kap):
                ck0 = covmat(theta0, theta0, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k], diag_only=True)
                cki = covmat(theta0, thetai, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k])
                ck = covmat(theta0, theta, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k])

                Ckiinv = Ciinvhs[k]
                Rkinvh = Rinvhs[k]

                ck_Ckinv_ck = (cki @ (Ckiinv @ Ckiinv.T - Rkinvh @ Rkinvh.T) @ cki.T).diag()

                Ckinv = Delta_inv_diags[k].diag() - QRinvhs[k] @ QRinvhs[k].T
                ck_Ckinv_Vkh = (ck @ Ckinv) * V[k].sqrt()

                predcov_g[k] = (ck0 - ck_Ckinv_ck) + \
                               (ck_Ckinv_Vkh**2).sum(1)

                term1[k] = (ck0 - ck_Ckinv_ck)
                term2[k] = (ck_Ckinv_Vkh**2).sum(1)
            for i in range(n0):
                predcov[:, :, i] = txPhi * predcov_g[:, i] @ txPhi.T
        #         torch.exp(lsigma2) * torch.eye(m) + \

            self.GPvarterm1 = term1
            self.GPvarterm2 = term2
        return predcov

    def predictvar(self, theta0):
        predcov = self.predictcov(theta0)

        n0 = theta0.shape[0]
        m = self.m
        predvar = torch.zeros(m, n0)
        for i in range(n0):
            predvar[:, i] = predcov[:, :, i].diag()
        return predvar

    def predictaddvar(self):
        _, lsigma2, _, _ = self.parameter_clamp(self.lLmb, self.lsigma2, self.lnugGPs, self.ltau2GPs)

        predictaddvar = (lsigma2.exp() * self.Fstd ** 2).squeeze(1)
        return predictaddvar

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

    
    def dss(self, theta0, f0, use_diag=False):
        """
        Returns the Dawid-Sebastani score averaged across test points.

        :param theta0:
        :param f0:
        :return:
        """
        __score_single = self.__dss_single
        if use_diag:
            __score_single = self.__dss_single_diag

        predmean = self.predictmean(theta0)
        predcov = self.predictcov(theta0)

        n0 = theta0.shape[0]

        score = 0
        for i in range(n0):
            score += __score_single(f0[:, i], predmean[:, i], predcov[:, :, i])
        score /= n0

        return score

    def chi2mean(self, theta0, f0):
        predmean = self.predictmean(theta0)
        predcov = self.predictcov(theta0)

        n0 = theta0.shape[0]

        chi2arr = torch.zeros(n0)

        chi2 = 0
        for i in range(n0):
            chi2arr[i] = self.__single_chi2mean(f0[:, i], predmean[:, i], predcov[:, :, i])
            chi2 += self.__single_chi2mean(f0[:, i], predmean[:, i], predcov[:, :, i])
        chi2 /= n0

        return chi2arr

    @staticmethod
    def __dss_single_diag(f, mu, Sigma):
        r = f - mu
        diagV = Sigma.diag()
        score_single = torch.log(diagV).sum() + (r * r / diagV).sum()
        # print('mse {:.6f}'.format((r**2).mean()))
        # print('diag cov mean: {:.6f}'.format(diagV.mean()))
        return score_single

    @staticmethod
    def __dss_single(f, mu, Sigma):  # Dawid-Sebastani score
        r = f - mu
        W, U = torch.linalg.eigh(Sigma)
        r_Sinvh = r @ U * 1 / torch.sqrt(W)

        score_single = torch.linalg.slogdet(Sigma).logabsdet + (r_Sinvh ** 2).sum()
        return score_single

    @staticmethod
    def __single_chi2mean(f, mu, Sigma):
        r = f - mu
        diagV = torch.diag(Sigma)

        return (torch.square(r / torch.sqrt(diagV))).mean()

    @staticmethod
    def parameter_clamp(lLmb, lsigma2, lnugs, ltau2s):
        # clamping
        lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5, 2.5)))).T
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, 1)))
        lnugs = parameter_clamping(lnugs, torch.tensor((-16, -8)))
        ltau2s = parameter_clamping(ltau2s, torch.tensor((-4, 4)))

        return lLmb, lsigma2, lnugs, ltau2s

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
    def __PCs(F, kap=None, threshold=0.9999):
        m, n = F.shape

        Phi, S, _ = torch.linalg.svd(F, full_matrices=False)
        v = (S ** 2).cumsum(0) / (S ** 2).sum()

        if kap is None:
            kap = int(torch.argwhere(v > threshold)[0][0] + 1)
            # if kap > 1:
            #     kap -= 1

        assert Phi.shape[1] == min(m, n)
        Phi = Phi[:, :kap]
        S = S[:kap]

        pcw = ((S ** 2).abs() / n).sqrt()

        G = (Phi / pcw).T @ F  # kap x n
        return G, Phi, pcw, kap
