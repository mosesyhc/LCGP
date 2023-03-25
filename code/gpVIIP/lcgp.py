import torch
import torch.nn as nn
import torch.jit as jit
from matern_covmat import covmat
from likelihood import negloglik_gp
from hyperparameter_tuning import parameter_clamping
from optim_elbo import optim_elbo_lbfgs, optim_elbo_adam, optim_elbo_qhadam
torch.set_default_dtype(torch.double)

JIT = False
if JIT:
    Module = jit.ScriptModule
else:
    Module = nn.Module

class LCGP(Module):
    def __init__(self, F, x,
                 Phi=None, kap=None, pcthreshold=0.9999,
                 clamping=True):
        """
        :param Phi:
        :param F:
        :param x:
        """
        super().__init__()
        self.method = 'LCGP'
        self.clamping = clamping
        self.p, self.n = F.shape
        self.F = F
        self.standardize_F()
        self.init_standard_x(x)  # standardize x to unit hypercube
        self.verify_dim(f=self.F, x=self.x)

        if Phi is None:
            self.G, self.pct, self.pcti, self.pcw, \
                self.D, self.kap = self.__PCs(F=self.F, kap=kap, threshold=pcthreshold)
        else:
            self.pct = Phi / torch.sqrt(torch.tensor(self.n,))
            self.pcti = Phi * torch.sqrt(torch.tensor(self.n,))
            self.D = (self.pcti ** 2).sum(0)
            self.G = self.pcti.T @ self.F
            self.kap = kap
            self.pcw = torch.ones(kap)
        # report recovery MSE
        Frawhat = self.tx_F((1 / self.D * self.pcti) @ self.G)
        Phi_mse = ((Frawhat - self.Fraw) ** 2).mean()
        print('#PCs: {:d}, recovery mse: {:.3E}'.format(self.kap, Phi_mse.item()))

        self.M = torch.zeros(self.kap, self.n)
        self.V = torch.full_like(self.M, torch.nan)
        self.Cinvhs = torch.full((self.kap, self.n, self.n), torch.nan)

        # set initial parameters
        self.init_params()
        # fill M and V
        self.compute_MV()
        self.buildtime: float = 0.0

    def init_params(self):
        x = self.x
        self.d = x.shape[1]
        llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([x.shape[1]])) +
                            torch.log(torch.std(x, 0)))
        # llmb = torch.cat((llmb, torch.Tensor([0])))
        lLmb = llmb.repeat(self.kap, 1)
        # lLmb[:, -1] = torch.log(torch.var(self.G, 1))

        self.lLmb = nn.Parameter(lLmb) #, requires_grad=False)
        self.lnugGPs = nn.Parameter(torch.Tensor(-14 * torch.ones(self.kap))) #, requires_grad=False)
        self.ltau2GPs = nn.Parameter(torch.Tensor(torch.zeros(self.kap))) #, requires_grad=False)

        Fhat = (self.pcti / self.D) @ self.G
        lsigma2 = torch.max(torch.log(((Fhat - self.F) ** 2).mean()),
                            torch.log(0.01 * (self.F ** 2).mean())) # torch.log(torch.tensor(0.25,))#
        self.lmse0 = torch.log(((Fhat - self.F) ** 2).mean())
        self.lsigma2reg = lsigma2.item()
        self.lsigma2 = nn.Parameter(lsigma2)  # !!!
        return

    def set_lsigma2(self, lsigma2):
        self.lsigma2 = nn.Parameter(lsigma2)

    def forward(self, x0):
        if x0.ndim < 2:
            x0 = x0.unsqueeze(1)

        lLmb, lsigma2, lnugGPs, ltau2GPs = self.get_param()

        M = self.M
        Cinvhs = self.Cinvhs
        x = self.x
        x0 = self.standardize_x(x0)

        kap = self.kap

        n0 = x0.shape[0]
        ghat = torch.zeros(kap, n0)
        for k in range(kap):
            ck = covmat(x0, x, llmb=lLmb[k], lnug=lnugGPs[k], ltau2=ltau2GPs[k])

            Ckinvh = Cinvhs[k]
            Ckinv_Mk = Ckinvh @ Ckinvh.T @ M[k]

            ghat[k] = ck @ Ckinv_Mk

        fhat = (self.pcti / self.D) @ ghat
        fhat = self.tx_F(fhat)
        return fhat  # , ghat

    def negpost(self):  #
        lLmb, lsigma2, lnugGPs, ltau2GPs = self.get_param()

        x = self.x

        n = self.n
        p = self.p
        kap = self.kap
        F = self.F
        D = self.D

        b = (self.G.T / D).T  # this is (pcti / D) @ F

        sigma2 = lsigma2.exp()

        negpost = 0
        for k in range(kap):
            Ck = covmat(x, x, llmb=lLmb[k], lnug=lnugGPs[k], ltau2=ltau2GPs[k])

            Wk_C, Uk_C = torch.linalg.eigh(Ck)
            Ckinvh = Uk_C / Wk_C.sqrt()

            Ak = 1 / D[k] * torch.eye(n) + sigma2 * Ckinvh @ Ckinvh.T

            Wk, Uk = torch.linalg.eigh(Ak)
            Akinvh = Uk / Wk.sqrt()
            Akinvhbk = Akinvh.T @ b[k]

            negpost += 1/2 * (torch.sum(torch.log(Wk.abs())) + torch.sum(torch.log(Wk_C.abs())))
            negpost -= 1/(2 * sigma2) * (Akinvhbk ** 2).sum()

        negpost += 1/(2 * sigma2) * (F ** 2).sum()
        negpost += n * (p + kap) / 2 * lsigma2
        return negpost

    def negprofilepost(self):
        lLmb, lsigma2, lnugGPs, ltau2GPs = self.get_param()

        F = self.F
        x = self.x

        p = self.p
        n = self.n
        kap = self.kap

        G = self.G

        negpost = 0
        for k in range(kap):
            negloggp_k, Cinvkdiag = negloglik_gp(llmb=lLmb[k], lnug=lnugGPs[k], ltau2=ltau2GPs[k], x=x, g=G[k])
            negpost += negloggp_k

        residF = F - (self.pcti / self.D) @ G  # unchanging
        negpost += n * p / 2 * lsigma2
        negpost += 1 / (2 * lsigma2.exp()) * (residF ** 2).sum()

        negpost += 1 / 2 * (lsigma2 - self.lsigma2reg) ** 2

        return negpost


    def negelbo(self):
        lLmb, lsigma2, lnugGPs, ltau2GPs = self.get_param()

        F = self.F
        x = self.x

        p = self.p
        n = self.n
        kap = self.kap

        M = self.M
        V = self.V

        D = self.D

        negelbo = 0
        for k in range(kap):
            negloggp_k, Cinvkdiag = negloglik_gp(llmb=lLmb[k], lnug=lnugGPs[k], ltau2=ltau2GPs[k], x=x, g=M[k])
            negelbo += negloggp_k
            negelbo += 1 / 2 * (Cinvkdiag * V[k]).sum()

        residF = F - (self.pcti / D) @ M
        negelbo += p * n / 2 * lsigma2
        negelbo += 1 / (2 * lsigma2.exp()) * (residF ** 2).sum()
        negelbo -= 1 / 2 * torch.log(V).sum()
        negelbo += 1 / (2 * lsigma2.exp()) * (V.T / D).sum()

        negelbo += 1 / 2 * (lsigma2 - self.lsigma2reg) ** 2

        return negelbo

    @torch.no_grad()
    def compute_MV(self):
        lLmb, lsigma2, lnugGPs, ltau2GPs = self.get_param()

        kap = self.kap
        x = self.x

        n = self.n

        G = self.G
        D = self.D

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)

        sigma2 = torch.exp(lsigma2)
        Cinvhs = torch.zeros(self.kap, self.n, self.n)
        for k in range(kap):
            C_k = covmat(x, x, llmb=lLmb[k], lnug=lnugGPs[k], ltau2=ltau2GPs[k])

            W_k, U_k = torch.linalg.eigh(C_k)

            Ckinvh = U_k / W_k.sqrt()
            # with torch.no_grad():
            M[k] = torch.linalg.solve(1 / D[k] * torch.eye(n) + sigma2 * Ckinvh @ Ckinvh.T, G[k] / D[k])  # D[k]
            V[k] = sigma2 * (1 / (1 / D[k] + sigma2 * (Ckinvh ** 2).sum(1)))

            Cinvhs[k] = Ckinvh
        self.M = M
        self.V = V
        self.Cinvhs = Cinvhs

    @torch.no_grad()
    def compute_fullcovG(self):
        kap, n = self.kap, self.n
        _, lsigma2, _, _ = self.get_param()

        D = self.D
        Cinvhs = self.Cinvhs

        covG = torch.zeros((kap, n, n))
        for k in range(kap):
            covG[k] = lsigma2.exp() * \
                      torch.linalg.inv(1 / D[k] * torch.eye(n) +
                                       lsigma2.exp() * Cinvhs[k] @ Cinvhs[k].T)
        return covG

    # @torch.no_grad()
    def get_param(self):
        if self.clamping:
            lLmb, lsigma2, lnugGPs, ltau2GPs = self.parameter_clamp(self.lLmb, self.lsigma2, self.lnugGPs, self.ltau2GPs)
        else:
            lLmb, lsigma2, lnugGPs, ltau2GPs = self.lLmb, self.lsigma2, self.lnugGPs, self.ltau2GPs
        return lLmb, lsigma2, lnugGPs, ltau2GPs

    @torch.no_grad()
    def get_param_grad(self):
        grad = []
        for p in filter(lambda p: p.requires_grad, self.parameters()):
            view = p.grad.data.view(-1)
            grad.append(view)
        grad = torch.cat(grad, 0)
        return grad

    def predictmean(self, x0):
        with torch.no_grad():
            self.compute_MV()
            fhat = self.forward(x0)
        return fhat

    @torch.no_grad()
    def predictvar(self, x0):
        self.compute_MV()
        x = self.x
        x0 = self.standardize_x(x0)

        lLmb, lsigma2, lnugGPs, ltau2GPs = self.get_param()

        txPhi = (self.pcti / self.D) * self.Fstd
        V = self.V

        n0 = x0.shape[0]
        kap = self.kap
        m = self.p

        Cinvhs = self.Cinvhs

        predcov = torch.zeros(m, m, n0)
        predcov_g = torch.zeros(kap, n0)

        term1 = torch.zeros(kap, n0)
        term2 = torch.zeros(kap, n0)
        for k in range(kap):
            ck0 = covmat(x0, x0, llmb=lLmb[k], lnug=lnugGPs[k], ltau2=ltau2GPs[k], diag_only=True)
            ck = covmat(x0, x, llmb=lLmb[k], lnug=lnugGPs[k], ltau2=ltau2GPs[k])
            Ckinvh = Cinvhs[k]

            ck_Ckinvh = ck @ Ckinvh
            ck_Ckinv_Vkh = ck @ Ckinvh @ Ckinvh.T * torch.sqrt(V[k])

            predcov_g[k] = (ck0 - (ck_Ckinvh ** 2).sum(1)) \
                            + (ck_Ckinv_Vkh ** 2).sum(1)

            term1[k] = (ck0 - (ck_Ckinvh ** 2).sum(1))
            term2[k] = (ck_Ckinv_Vkh**2).sum(1)

        predvar = (predcov_g.T @ (txPhi ** 2).T).T
        # for i in range(n0):
        #     predcov[:, :, i] = txPhi * predcov_g[:, i] @ txPhi.T
        return_cov = True
        if return_cov:
            CH = predcov_g.T.sqrt()[:, :, None] * txPhi.T[None, :, :]

        self.GPvarterm1 = term1
        self.GPvarterm2 = term2
        return predvar
    #
    # def predictvar(self, x0):
    #     predcov = self.predictcov(x0)
    #
    #     n0 = x0.shape[0]
    #     p = self.p
    #     predvar = torch.zeros(p, n0)
    #     for i in range(n0):
    #         predvar[:, i] = predcov[:, :, i].diag()
    #     return predvar

    def predictaddvar(self):
        lLmb, lsigma2, lnugGPs, ltau2GPs = self.get_param()
        predictaddvar = (lsigma2.exp() * self.Fstd ** 2).squeeze(1)
        return predictaddvar

    def test_mse(self, x0, f0):
        with torch.no_grad():
            m, n = f0.shape
            fhat = self.predictmean(x0)
            return ((fhat - f0) ** 2).sum() / (m * n)

    def test_rmse(self, x0, f0):
        return torch.sqrt(self.test_mse(x0=x0, f0=f0))

    def test_individual_error(self, x0, f0):
        with torch.no_grad():
            fhat = self.predictmean(x0)
            return torch.sqrt((fhat - f0) ** 2).mean(0)

    def init_standard_x(self, x):
        if x.ndim < 2:
            x = x.unsqueeze(1)
        self.x_orig = x.clone()
        self.x_max = x.max(0).values
        self.x_min = x.min(0).values
        self.x = (x - self.x_min) / (self.x_max - self.x_min)

    def standardize_x(self, x0):
        return (x0 - self.x_min) / (self.x_max - self.x_min)

    def standardize_F(self):
        F = self.F
        self.Fraw = F.clone()
        self.Fmean = F.mean(1).unsqueeze(1)
        self.Fstd = F.std(1).unsqueeze(1) # / torch.sqrt(torch.tensor(self.n / (self.n-1),))
        self.F = (F - self.Fmean) / self.Fstd

    def tx_x(self, xs):
        return xs * (self.x_max - self.x_min) + self.x_min

    def tx_F(self, Fs):
        return Fs * self.Fstd + self.Fmean

    def dss(self, x0, f0, use_diag=False):
        """
        Returns the Dawid-Sebastani score averaged across test points.

        :param x0:
        :param f0:
        :return:
        """
        __score_single = self.__dss_single
        if use_diag:
            __score_single = self.__dss_single_diag

        predmean = self.predictmean(x0)
        predcov = self.predictcov(x0)

        n0 = x0.shape[0]

        score = 0
        for i in range(n0):
            score += __score_single(f0[:, i], predmean[:, i], predcov[:, :, i])
        score /= n0

        return score

    def chi2mean(self, x0, f0):

        predmean = self.predictmean(x0)
        predcov = self.predictcov(x0)

        n0 = x0.shape[0]

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
        d = torch.tensor(lLmb.shape[1],)
        # lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5 + 1/2 * torch.log(d), 2.5)))).T  # + 1/2 * log dimension
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, 1)))
        # lnugs = parameter_clamping(lnugs, torch.tensor((-16, -8)))
        # ltau2s = parameter_clamping(ltau2s, torch.tensor((-4, 4)))

        return lLmb, lsigma2, lnugs, ltau2s

    @staticmethod
    def __PCs(F, kap=None, threshold=0.9999):
        m, n = F.shape

        Phi, S, _ = torch.linalg.svd(F, full_matrices=False)
        v = (S ** 2).cumsum(0) / (S ** 2).sum()

        if kap is None:
            kap = int(torch.argwhere(v > threshold)[0][0] + 1)

        assert Phi.shape[1] == min(m, n)
        S = S[:kap]

        pcw = ((S ** 2).abs()).sqrt()
        pct = Phi[:, :kap] * pcw / torch.sqrt(torch.tensor(n,))
        pcti = Phi[:, :kap] * torch.sqrt(torch.tensor(n,)) / pcw

        D = (pcti ** 2).sum(0)

        G = pcti.T @ F  # kap x n
        return G, pct, pcti, pcw, D, kap

    def fit(self, verbose=False):
        niter, flag = self.fit_bfgs(verbose=verbose)
        return niter, flag

    def fit_bfgs(self, **kwargs):
        _, niter, flag = optim_elbo_lbfgs(self, **kwargs)
        return niter, flag

    def fit_adam(self, maxiter=100, **kwargs):
        _, niter, flag = optim_elbo_adam(self, maxiter=maxiter, **kwargs)
        return niter, flag

    def fit_qhadam(self, maxiter=100, **kwargs):
        _, niter, flag = optim_elbo_qhadam(self, maxiter=maxiter, **kwargs)
        return niter, flag

    def verify_dim(self, f, x):
        m, nf = f.shape
        nx, d = x.shape

        if nf != nx:
            raise ValueError('Number of inputs (x) differs from number of outputs (f), '
                             'f.shape[1] != x.shape[0]')