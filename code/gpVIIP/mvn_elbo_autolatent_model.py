import torch
import torch.nn as nn
import torch.jit as jit
from matern_covmat import cormat
from likelihood import negloglik_gp
from prediction import pred_gp
from hyperparameter_tuning import parameter_clamping


class MVN_elbo_autolatent(jit.ScriptModule):
    def __init__(self, F, theta,
                 Phi=None, kap=None, pcthreshold=0.99,
                 lLmb=None, lsigma2=None,
                 initlLmb=True, initlsigma2=True,
                 clamping=True):
        """

        :param Phi:
        :param F:
        :param theta:
        :param lLmb:
        :param lsigma2:
        :param initlLmb:
        :param initlsigma2:
        """
        super().__init__()
        self.method = 'MVGP'
        self.clamping = clamping
        self.m, self.n = F.shape
        self.F = F

        self.standardize_F()

        if Phi is None:
            self.G, self.Phi, self.pcw, self.kap = self.__PCs(F=self.F, kap=kap, threshold=pcthreshold)

            Fhat = self.tx_F((self.Phi * self.pcw) @ self.G)
            Phi_mse = ((Fhat - self.Fraw) ** 2).mean()
            print('#PCs: {:d}, recovery mse: {:.3E}'.format(self.kap, Phi_mse.item()))
        else:
            self.G = Phi.T @ self.F
            self.Phi = Phi
            self.kap = kap
            self.pcw = torch.ones(kap)

        self.M = torch.zeros(self.kap, self.n)
        self.V = torch.full_like(self.M, torch.nan)
        self.Cinvhs = torch.full((self.kap, self.n, self.n), torch.nan)

        self.theta = theta
        self.d = theta.shape[1]
        if initlLmb or lLmb is None:
            llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                                torch.log(torch.std(theta, 0)))
            llmb = torch.cat((llmb, torch.Tensor([0])))
            lLmb = llmb.repeat(self.kap, 1)
            lLmb[:, -1] = torch.log(torch.var(self.G, 1))

        self.lLmbreg = lLmb.clone()
        self.lLmb = nn.Parameter(lLmb)

        lnugGPs = torch.Tensor(-16 * torch.ones(self.kap))
        self.lnugGPs = nn.Parameter(lnugGPs)

        if initlsigma2 or lsigma2 is None:
            lsigma2 = torch.log(((self.Phi @ self.Phi.T @ self.F - self.F) ** 2).mean())
        self.lmse0 = lsigma2.item()
        self.lsigma2 = nn.Parameter(lsigma2)
        self.buildtime: float = 0.0

    @jit.script_method
    def forward(self, theta0):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lnugGPs = self.lnugGPs

        if self.clamping:
            lLmb, lsigma2, lnugGPs = self.parameter_clamp(lLmb, lsigma2, lnugGPs)

        M = self.M
        Cinvhs = self.Cinvhs
        theta = self.theta

        kap = self.kap

        n0 = theta0.shape[0]
        ghat = torch.zeros(kap, n0)
        for k in range(kap):
            ck0 = cormat(theta0, theta, llmb=lLmb[k])
            nug = torch.exp(lnugGPs[k]) / (1 + torch.exp(lnugGPs[k]))
            ck = (1 - nug) * ck0

            Ckinvh = Cinvhs[k]
            Ckinv_Mk = Ckinvh @ Ckinvh.T @ M[k]

            ghat[k] = ck @ Ckinv_Mk

        fhat = (self.Phi * self.pcw) @ ghat
        fhat = self.tx_F(fhat)
        return fhat  # , ghat

    def negelbo(self):
        lLmb = self.lLmb
        lsigma2 = self.lsigma2
        lnugGPs = self.lnugGPs

        if self.clamping:
            lLmb, lsigma2, lnugGPs = self.parameter_clamp(lLmb, lsigma2, lnugGPs)

        F = self.F
        theta = self.theta

        m = self.m
        n = self.n
        kap = self.kap

        M = self.M
        V = self.V

        negelbo = 0
        for k in range(kap):
            negloggp_k = negloglik_gp(llmb=lLmb[k], lnug=lnugGPs[k], theta=theta, g=M[k])
            negelbo += negloggp_k

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

        n = self.n
        G = self.G
        sigma2 = torch.exp(lsigma2)

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)

        Cinvhs = torch.zeros(self.kap, self.n, self.n)  # Ukh @ Ukh.T == inv(Ck)
        sig2gps = torch.zeros(self.kap)

        for k in range(kap):
            C_k0 = cormat(theta, theta, lLmb[k])
            nugk = torch.exp(lnugGPs[k]) / (1 + torch.exp(lnugGPs[k]))
            C_k = (1 - nugk) * C_k0 + nugk * torch.eye(theta.shape[0])

            W_k, U_k = torch.linalg.eigh(C_k)

            Ckinvh = U_k / W_k.sqrt()
            CkinvhGk = Ckinvh.T @ G[k]
            sig2k = (n * (CkinvhGk ** 2).mean() + 10) / (n + 10)

            Mk = torch.linalg.solve(torch.eye(n) + sigma2 / sig2k * Ckinvh @ Ckinvh.T, G[k])

            M[k] = Mk
            V[k] = 1 / (1 / sigma2 + (Ckinvh ** 2 / sig2k).sum(1))

            # save
            Cinvhs[k] = Ckinvh
            sig2gps[k] = sig2k

        self.M = M
        self.V = V
        self.Cinvhs = Cinvhs
        self.sig2gps = sig2gps

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

            Cinvhs = self.Cinvhs
            sig2gps = self.sig2gps

            predcov = torch.zeros(m, m, n0)
            predcov_g = torch.zeros(kap, n0)
            for k in range(kap):
                ck0 = cormat(theta0, theta, llmb=lLmb[k])
                nug = torch.exp(lnugGPs[k]) / (1 + torch.exp(lnugGPs[k]))
                ck = (1 - nug) * ck0

                Ckinvh = Cinvhs[k]

                ck_Ckinvh = ck @ Ckinvh
                ck_Ckinv_Vkh = ck @ Ckinvh @ Ckinvh.T * torch.sqrt(V[k])

                predcov_g[k] = sig2gps[k] * nug * (1 - (ck_Ckinvh ** 2).sum(1)) + \
                                (ck_Ckinv_Vkh ** 2).sum(1)


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

    @staticmethod
    def __dss_single_diag(f, mu, Sigma):
        r = f - mu
        diagV = Sigma.diag()
        score_single = torch.log(diagV).sum() + (r * r / diagV).sum()
        # print('mse {:.6f}'.format((r**2).mean()))
        # print('diag cov mean: {:.6f}'.format(diagV.mean()))
        print('logdet: {:.6f}, quadratic: {:.6f}'.format(torch.log(diagV).sum(), (r * r / diagV).sum()))
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
    def parameter_clamp(lLmb, lsigma2, lnugs):
        # clamping
        lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5, 2.5)))).T
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, 0)))
        lnugs = parameter_clamping(lnugs, torch.tensor((-20, -8)))

        return lLmb, lsigma2, lnugs

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
