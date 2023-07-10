import torch
import torch.nn as nn
from matern_covmat import covmat
from likelihood import negloglik_gp
from hyperparameter_tuning import parameter_clamping
from optim_elbo import optim_elbo_lbfgs, optim_elbo_adam, optim_elbo_qhadam
torch.set_default_dtype(torch.double)


class LCGP(nn.Module):
    def __init__(self,
                 y: torch.double,
                 x: torch.double,
                 q: int = None,
                 var_threshold: float = None,
                 error_structure: str = 'diagonal',
                 param_clamp=True):
        super().__init__()
        self.method = 'LCGP'
        self.x = x

        self.y_orig, self.ymean, self.y = self.center_y(y)

        self.param_clamp = param_clamp
        if (q is not None) and (var_threshold is not None):
            raise ValueError('Include only q or var_threshold but not both.')
        self.q = q
        self.var_threshold = var_threshold

        # placeholders for variables
        self.n, self.d, self.p = 0, 0, 0
        # verify that input and output dimensions match
        self.verify_dim(y, x)

        self.x_orig, self.x_max, self.x_min = (x.clone(), torch.zeros(size=([self.d]), dtype=torch.double),
                                               torch.zeros(size=([self.d]), dtype=torch.double))

        # standardize x to unit hypercube
        self.init_standard_x(x)

        # reset q if none is provided
        self.g, self.phi, self.diag_D, self.q = self.init_phi(var_threshold=var_threshold)

        self.lLmb, self.lLmb0, \
            self.lnugGP, self.lsigma2s = (torch.zeros(size=[self.q, self.d], dtype=torch.double),
                                          torch.zeros(size=[self.q], dtype=torch.double),
                                          torch.zeros(size=[self.q], dtype=torch.double),
                                          torch.zeros(size=[self.p], dtype=torch.double))

        self.init_params()

        # placeholders for predictive quantities
        self.CinvMs = torch.zeros(size=[self.q, self.n])
        self.Ths = torch.zeros(size=[self.q, self.n, self.n])

        # yhat0 = (self.phi / self.lsigma2s.exp().sqrt() @ self.g).T  # diagonal or scalar
        # self.yhat0 = yhat0
        # print((yhat0 - y)**2).mean()

    def init_phi(self, var_threshold: float = None):
        y, q = self.y, self.q
        n, p = self.n, self.p

        left_u, singvals, _ = torch.linalg.svd(y.T, full_matrices=False)

        # no reduction
        if (q is None) and (var_threshold is None):
            q = p
        elif (q is None) and (var_threshold is not None):
            cumvar = (singvals ** 2).cumsum(0) / (singvals ** 2).sum()
            q = int(torch.argwhere(cumvar > var_threshold)[0][0] + 1)

        assert left_u.shape[1] == min(n, p)
        singvals = singvals[:q]

        # singvals_abs = singvals.abs()
        phi = left_u[:, :q] * torch.sqrt(torch.tensor(n,)) / singvals
        diag_D = (phi ** 2).sum(0)

        g = phi.T @ y.T
        return g, phi, diag_D, q

    def init_params(self):
        x = self.x
        d = self.d

        llmb = 0.5 * torch.log(torch.Tensor([d])) + torch.log(torch.std(x, 0))
        lLmb = llmb.repeat(self.q, 1)
        lLmb0 = torch.zeros(self.q)
        lnugGPs = torch.Tensor(-14 * torch.ones(self.q))

        lsigma2_diag = torch.Tensor(torch.log(self.y.var(0)))

        self.lLmb = nn.Parameter(lLmb)
        self.lLmb0 = nn.Parameter(lLmb0)
        self.lnugGPs = nn.Parameter(lnugGPs)
        self.lsigma2s = nn.Parameter(lsigma2_diag)
        return

    def verify_dim(self, y, x):
        ny, p = y.shape
        nx, d = x.shape

        if ny != nx:
            raise ValueError('Number of inputs (x) differs from number of outputs (y), '
                             'y.shape[0] != x.shape[0]')
        else:
            self.n = nx
            self.d = d
            self.p = p
            return

    def init_standard_x(self, x):
        if x.ndim < 2:
            x = x.unsqueeze(1)
        # self.x_orig = x.clone()
        self.x_max = x.max(0).values
        self.x_min = x.min(0).values
        self.x = (x - self.x_min) / (self.x_max - self.x_min)
        return

    def forward(self, x0):
        return self.predict(x0)

    @torch.no_grad()
    def predict(self, x0):
        x = self.x
        lLmb = self.lLmb
        lLmb0 = self.lLmb0
        lnugGP = self.lnugGP

        lsigma2 = self.lsigma2s
        phi = self.phi

        psi = (phi.T * lsigma2.exp().sqrt()).T

        CinvM = self.CinvMs
        Th = self.Ths

        if x0.ndim < 2:
            x0 = x0.unsqueeze(1)
        x0 = self.standardize_x(x0)
        n0 = x0.shape[0]

        ghat = torch.zeros([self.q, n0])
        gvar = torch.zeros([self.q, n0])
        for k in range(self.q):
            c00k = covmat(x0, x0, diag_only=True, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGP[k])
            c0k = covmat(x0, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGP[k])

            ghat[k] = c0k @ CinvM[k]
            gvar[k] = c00k - ((c0k @ Th[k]) ** 2).sum(1)

        predmean = (psi @ ghat).T + self.ymean
        predvar = gvar.T @ (psi ** 2).T
        return predmean, predvar

    @torch.no_grad()
    def compute_aux_predictive_quantities(self):
        x = self.x
        lLmb = self.lLmb
        lLmb0 = self.lLmb0
        lnugGP = self.lnugGP
        lsigma2 = self.lsigma2s

        D = self.diag_D
        # B := Y @ Sigma^{-1/2} @ Phi
        B = (self.y / lsigma2.exp().sqrt()) @ self.phi

        CinvM = torch.zeros([self.q, self.n])
        Th = torch.zeros([self.q, self.n, self.n])

        for k in range(self.q):
            Ck = covmat(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGP[k])

            Wk, Uk = torch.linalg.eigh(Ck)

            # (I + D_k * C_k)^{-1}
            IpdkCkinv = Uk / (1.0 + D[k] * Wk) @ Uk.T

            CkinvMk = IpdkCkinv @ B.T[k]
            Thk = Uk * ((D[k] * Wk**2) / (Wk**2 + D[k] * Wk**3)).sqrt() @ Uk.T

            CinvM[k] = CkinvMk
            Th[k] = Thk
        self.CinvMs = CinvM
        self.Ths = Th

    def standardize_x(self, x0):
        return (x0 - self.x_min) / (self.x_max - self.x_min)

    def center_y(self, y):
        ymean = y.mean(0)

        return y.clone(), ymean, y - ymean

class LCGP_homogeneous_error(LCGP):
    def __init__(self, y, x):
        super(LCGP_homogeneous_error, self).__init__(y, x)


class LCGP_diagonal_error(LCGP):
    def __init__(self, y, x):
        super(LCGP_diagonal_error, self).__init__(y, x)


class LCGP_block_error(LCGP):
    def __init__(self, y, x):
        super(LCGP_block_error, self).__init__(y, x)


class LCGP_heterogeneous_error(LCGP):
    def __init__(self, y, x):
        super(LCGP_heterogeneous_error, self).__init__(y, x)


class LCGP_old(nn.Module):
    def __init__(self, Y, x, kap, pcthreshold=0.9999,
                 Phi=None,
                 clamping=True):
        super().__init__()
        self.method = 'LCGP'
        self.clamping = clamping
        self.p, self.n = Y.shape
        self.Y = Y
        self.standardize_Y()
        self.init_standard_x(x)  # standardize x to unit hypercube
        self.verify_dim(Y=self.Y, x=self.x)

        if Phi is None:
            self.G, self.pct, self.pcti, self.pcw, \
                self.D, self.kap = self.__PCs(Y=self.Y, kap=kap, threshold=pcthreshold)
        else:
            self.pct = Phi / torch.sqrt(torch.tensor(self.n,))
            self.pcti = Phi * torch.sqrt(torch.tensor(self.n,))
            self.D = (self.pcti ** 2).sum(0)
            self.G = self.pcti.T @ self.Y
            self.kap = kap
            self.pcw = torch.ones(kap)
        # report recovery MSE
        Frawhat = self.tx_Y((1 / self.D * self.pcti) @ self.G)
        Phi_mse = ((Frawhat - self.Yraw) ** 2).mean()
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

        self.lLmb = nn.Parameter(lLmb)  #, requires_grad=False)
        self.lnugGPs = nn.Parameter(torch.Tensor(-14 * torch.ones(self.kap)))  #, requires_grad=False)
        self.ltau2GPs = nn.Parameter(torch.Tensor(torch.zeros(self.kap)))  #, requires_grad=False)

        Fhat = (self.pcti / self.D) @ self.G
        lsigma2 = torch.max(torch.log(((Fhat - self.Y) ** 2).mean()),
                            torch.log(0.01 * (self.Y ** 2).mean())) # torch.log(torch.tensor(0.25,))#
        self.lmse0 = torch.log(((Fhat - self.Y) ** 2).mean())
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
            ck = covmat(x0, x, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k])

            Ckinvh = Cinvhs[k]
            Ckinv_Mk = Ckinvh @ Ckinvh.T @ M[k]

            ghat[k] = ck @ Ckinv_Mk

        fhat = (self.pcti / self.D) @ ghat
        fhat = self.tx_Y(fhat)
        return fhat  # , ghat

    def negpost(self):  #
        lLmb, lsigma2, lnugGPs, ltau2GPs = self.get_param()

        x = self.x

        n = self.n
        p = self.p
        kap = self.kap
        F = self.Y
        D = self.D

        b = (self.G.T / D).T  # this is (pcti / D) @ F

        sigma2 = lsigma2.exp()

        negpost = 0
        for k in range(kap):
            Ck = covmat(x, x, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k])

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

        F = self.Y
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

        F = self.Y
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
            C_k = covmat(x, x, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k])

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
            ck0 = covmat(x0, x0, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k], diag_only=True)
            ck = covmat(x0, x, llmb=lLmb[k], llmb0=ltau2GPs[k], lnug=lnugGPs[k])
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

    def standardize_Y(self):
        Y = self.Y
        self.Yraw = Y.clone()
        self.Ymean = Y.mean(1).unsqueeze(1)
        self.Fstd = Y.std(1).unsqueeze(1)  # / torch.sqrt(torch.tensor(self.n / (self.n-1),))
        self.Y = (Y - self.Ymean) / self.Fstd

    def tx_x(self, xs):
        return xs * (self.x_max - self.x_min) + self.x_min

    def tx_Y(self, Ys):
        return Ys * self.Fstd + self.Ymean

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
    def __PCs(Y, kap=None, threshold=0.9999):
        m, n = Y.shape

        Phi, S, _ = torch.linalg.svd(Y, full_matrices=False)
        v = (S ** 2).cumsum(0) / (S ** 2).sum()

        if kap is None:
            kap = int(torch.argwhere(v > threshold)[0][0] + 1)

        assert Phi.shape[1] == min(m, n)
        S = S[:kap]

        pcw = ((S ** 2).abs()).sqrt()
        pct = Phi[:, :kap] * pcw / torch.sqrt(torch.tensor(n,))
        pcti = Phi[:, :kap] * torch.sqrt(torch.tensor(n,)) / pcw

        D = (pcti ** 2).sum(0)

        G = pcti.T @ Y  # kap x n
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

    def verify_dim(self, Y, x):
        m, nf = Y.shape
        nx, d = x.shape

        if nf != nx:
            raise ValueError('Number of inputs (x) differs from number of outputs (f), '
                             'f.shape[1] != x.shape[0]')