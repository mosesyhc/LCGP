import torch
import torch.nn as nn
import torch.jit as jit
from matern_covmat import cormat
from likelihood import negloglik_gp
from prediction import pred_gp
from hyperparameter_tuning import parameter_clamping


class MVN_elbo_autolatent(jit.ScriptModule):
    def __init__(self, Phi, F, theta,
                 lLmb=None, lsigma2=None,
                 initlLmb=True, initlsigma2=True):
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
        self.kap = Phi.shape[1]
        self.m, self.n = F.shape
        self.M = torch.zeros(self.kap, self.n)
        self.Phi = Phi

        self.F = F

        self.theta = theta
        self.d = theta.shape[1]
        if initlLmb or lLmb is None:
            llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                               torch.log(torch.std(theta, 0)))
            llmb = torch.cat((llmb, torch.Tensor([0])))
            lLmb = llmb.repeat(self.kap, 1)
            lLmb[:, -1] = torch.log(torch.var(Phi.T @ self.F, 1))
        self.lLmb = nn.Parameter(lLmb)  # , requires_grad=False
        if initlsigma2 or lsigma2 is None:
            lsigma2 = torch.log(((Phi @ Phi.T @ self.F - self.F)**2).mean())
        self.lsigma2 = nn.Parameter(lsigma2)  # nn.Parameter(torch.tensor((-8,)), requires_grad=False)



    @jit.script_method
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
            ck = cormat(theta0, theta, llmb=lLmb[k])
            Ck = cormat(theta, theta, llmb=lLmb[k])

            Wk, Uk = torch.linalg.eigh(Ck)
            Ukh = Uk / torch.sqrt(Wk)
            Ckinv_Mk = Ukh @ Ukh.T @ M[k]

            ghat[k] = ck @ Ckinv_Mk
            # ghat[k], _ = pred_gp(llmb=lLmb[k], theta=theta, thetanew=theta0, g=M[k])

        fhat = Phi @ ghat
        # fhat = (fhat * self.Fstd) + self.Fmean
        return fhat, ghat

    def negelbo(self, lsigma2=None):
        lLmb = self.lLmb
        if lsigma2 is None:
            lsigma2 = self.lsigma2
        lLmb_clm, lsigma2_clm = self.parameter_clamp(lLmb, lsigma2)

        sigma2 = torch.exp(lsigma2_clm)
        Phi = self.Phi
        F = self.F
        theta = self.theta

        m = self.m
        n = self.n
        kap = self.kap

        M = self.M
        V = self.V

        negelbo = 0
        for k in range(kap):
            negloggp_k, _ = negloglik_gp(llmb=lLmb_clm[k], theta=theta, g=M[k])
            negelbo += negloggp_k

        residF = F - (Phi @ M)
        negelbo = m * n / 2 * lsigma2_clm
        negelbo += 1 / (2 * lsigma2_clm.exp()) * (residF ** 2).sum()
        negelbo -= 1/2 * torch.log(V).sum()

        return negelbo

    def compute_MV(self):
        lsigma2 = self.lsigma2
        lLmb = self.lLmb
        lLmb, lsigma2 = self.parameter_clamp(lLmb, lsigma2)

        kap = self.kap
        theta = self.theta

        n = self.n
        Phi = self.Phi
        F = self.F
        sigma2 = torch.exp(lsigma2)

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)
        for k in range(kap):
            C_k = cormat(theta, theta, lLmb[k])
            W_k, U_k = torch.linalg.eigh(C_k)
            Winv_k = 1 / W_k
            Mk = torch.linalg.solve(torch.eye(n) + sigma2 * U_k * Winv_k @ U_k.T, Phi[:, k] @ F)
            M[k] = Mk
            V[k] = 1 / (1 / sigma2 + torch.diag((U_k * Winv_k) @ U_k.T))
        self.M = M
        self.V = V

    def predictmean(self, theta0):
        with torch.no_grad():
            self.compute_MV()
            fhat = self.forward(theta0)
        return(fhat)

    def predictcov(self, theta0):
        with torch.no_grad():
            self.compute_MV()
            theta = self.theta
            lLmb = self.lLmb
            lsigma2 = self.lsigma2

            Phi = self.Phi
            V = self.V

            n0 = theta0.shape[0]
            kap = self.kap
            m = self.m

            predcov = torch.zeros(m, m, n0)

            predcov_g = torch.zeros(kap, n0)
            for k in range(kap):
                ck0 = cormat(theta0, theta0, llmb=lLmb[k], diag_only=True)
                ck = cormat(theta0, theta, llmb=lLmb[k])
                Ck = cormat(theta, theta, llmb=lLmb[k])

                Wk, Uk = torch.linalg.eigh(Ck)
                Ukh = Uk / torch.sqrt(Wk)

                Ckinvh_Vkh = Ukh * torch.sqrt(V[k])

                ck_Ckinvh = ck @ Ukh
                ck_Ckinv_Vkh = ck @ Ukh @ Ukh.T * torch.sqrt(V[k])

                predcov_g[k] = ck0 - (ck_Ckinvh**2).sum(1) + (ck_Ckinv_Vkh**2).sum(1)

            for i in range(n0):
                predcov[:, :, i] = torch.exp(lsigma2) * torch.eye(m) + Phi * predcov_g[:, i] @ Phi.T

        return predcov, predcov_g

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
            return ((fhat - f0) ** 2).sum() / (m*n)

    def test_rmse(self, theta0, f0):
        return torch.sqrt(self.test_mse(theta0=theta0, f0=f0))

    def test_individual_error(self, theta0, f0):
        with torch.no_grad():
            fhat = self.predictmean(theta0)
            return torch.sqrt((fhat - f0)**2).mean(0)

    def standardize_F(self):
        if self.F is not None:
            F = self.F
            self.Fraw = F.clone()
            self.Fmean = F.mean(1).unsqueeze(1)
            self.Fstd = F.std(1).unsqueeze(1)
            self.F = (F - self.Fmean) / self.Fstd

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

        # return (torch.abs(r / torch.sqrt(diagV)) > 1.65).sum() / r.shape[0]
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
    def parameter_clamp(lLmb, lsigma2):
        # clamping
        lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5, 2.5)))).T
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, -1)))

        return lLmb, lsigma2
