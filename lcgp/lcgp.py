import torch
import torch.nn as nn
from .covmat import Matern32
from .hyperparameter_tuning import parameter_clamping
from .optim import optim_lbfgs
torch.set_default_dtype(torch.double)


class LCGP(nn.Module):
    """
    Implementation of latent component Gaussian process.
    """
    def __init__(self,
                 y: torch.double,
                 x: torch.double,
                 q: int = None,
                 var_threshold: float = None,
                 parameter_clamp_flag: bool = False,
                 robust_mean: bool = True,
                 penalty_const: dict = None):
        """
        Constructor for LCGP class.

        :param y: Simulation outputs, of size (number of input, dimension of output).
        :param x: Inputs, of size (number of input, dimension of input).
        :param q: Number of latent components to construct.
        Defaults to dimension of output.
        :param var_threshold: Value between (0, 1).  Minimum portion of variance
        to be explained through singular value decomposition.  The number of latent
        components, `q`, is determined by the cumulative sum of the square of singular
        values first exceeding `var_threshold`.  Defaults to 1.
        :param parameter_clamp_flag: Set soft boundary for GP hyperparameters if True.
        Defaults to False.
        :param robust_mean: Set output standardization option to median and absolute
        deviation if True.
            Defaults to True.
        :param penalty_const: Dictionary to set regularization constants for
        log_lengthscale and log_scale, e.g.,
        {'lLmb': 10, 'lLmb0': 5}.  Defaults to {'lLmb': 40, 'lLmb0': 5}.
        """
        super().__init__()
        self.method = 'LCGP'
        self.x = x

        self.parameter_clamp_flag = parameter_clamp_flag
        if (q is not None) and (var_threshold is not None):
            raise ValueError('Include only q or var_threshold but not both.')
        self.q = q
        self.var_threshold = var_threshold

        # standardize x to unit hypercube
        self.x, self.x_min, self.x_max, self.x_orig, self.xnorm = \
            self.init_standard_x(x)
        # standardize y
        self.y, self.ymean, self.ystd, self.y_orig = self.standardize_y(y, robust_mean)

        # placeholders for variables
        self.n, self.d, self.p = 0, 0, 0
        # verify that input and output dimensions match
        self.verify_dim(self.y, self.x)

        # reset q if none is provided
        self.g, self.phi, self.diag_D, self.q = \
            self.init_phi(var_threshold=var_threshold)
        # self.ghat = torch.zeros_like(self.g)

        self.lLmb, self.lLmb0, self.lnugGPs, self.lsigma2s = \
            (torch.zeros(size=[self.q, self.d], dtype=torch.double),
             torch.zeros(size=[self.q], dtype=torch.double),
             torch.zeros(size=[self.q], dtype=torch.double),
             torch.zeros(size=[self.p], dtype=torch.double))

        if penalty_const is None:
            pc = {'lLmb': 40, 'lLmb0': 5}
        else:
            pc = penalty_const
            for k, v in pc.items():
                assert v >= 0, 'penalty constant should be nonnegative.'
        self.penalty_const = pc

        self.init_params()

        # placeholders for predictive quantities
        self.CinvMs = torch.zeros(size=[self.q, self.n])
        self.Ths = torch.zeros(size=[self.q, self.n, self.n])

    def init_phi(self, var_threshold: float = None):
        """
        Initialization of orthogonal basis, computed with singular value decomposition.
        """
        y, q = self.y, self.q
        n, p = self.n, self.p

        left_u, singvals, _ = torch.linalg.svd(y, full_matrices=False)

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

        g = phi.T @ y
        return g, phi, diag_D, q

    def init_params(self):
        """
        Initializes parameters for LCGP.
        """
        x = self.x
        d = self.d

        llmb = 0.5 * torch.log(torch.Tensor([d])) + torch.log(torch.std(x, 0))
        lLmb = llmb.repeat(self.q, 1)
        lLmb0 = torch.zeros(self.q)
        lnugGPs = torch.Tensor(-6 * torch.ones(self.q))

        lsigma2_diag = torch.Tensor(torch.log(self.y.var(1)))

        self.lLmb = nn.Parameter(lLmb)
        self.lLmb0 = nn.Parameter(lLmb0)
        self.lnugGPs = nn.Parameter(lnugGPs)
        self.lsigma2s = nn.Parameter(lsigma2_diag)
        return

    def verify_dim(self, y, x):
        """
        Verifies if input and output dimensions match.  Sets class variables for
        dimensions. Throws error if the dimensions do not match.
        """
        p, ny = y.shape
        nx, d = x.shape

        assert ny == nx, 'Number of inputs (x) differs from number of outputs (y), ' \
                         'y.shape[0] != x.shape[0]'

        self.n = nx
        self.d = d
        self.p = p
        return

    def fit(self, **kwargs):
        """
        Calls optimizer for fitting the LCGP instance.
        """
        _, niter, flag = optim_lbfgs(self, **kwargs)
        return niter, flag

    def forward(self, x0):
        """
        Returns predictive mean at new input `x0`.  The output is of size
        (number of new input, output dimension).
        """
        return self.predict(x0)[0]

    @torch.no_grad()
    def predict(self, x0, return_fullcov=False):
        """
        Returns predictive quantities at new input `x0`.  Both outputs are of
        size (number of new input, output dimension).
        :param x0: New input of size (number of new input, dimension of input).
        :param return_fullcov: Returns (predictive mean, predictive variance,
        variance for the true mean, full predictive covariance) if True.  Otherwise,
        only return the first three quantities.
        """
        x = self.x
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()

        phi = self.phi

        CinvM = self.CinvMs
        Th = self.Ths

        x0 = self.standardize_x(x0)
        n0 = x0.shape[0]

        ghat = torch.zeros([self.q, n0])
        gvar = torch.zeros([self.q, n0])
        for k in range(self.q):
            c00k = Matern32(x0, x0, diag_only=True, llmb=lLmb[k], llmb0=lLmb0[k],
                            lnug=lnugGPs[k])
            c0k = Matern32(x0, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

            ghat[k] = c0k @ CinvM[k]
            gvar[k] = c00k - ((c0k @ Th[k]) ** 2).sum(1)

        psi = (phi.T * lsigma2s.exp().sqrt()).T
        self.g = psi.T @ self.y
        self.ghat = (lsigma2s.exp().sqrt() * ghat.T).T

        predmean = psi @ ghat
        confvar = (gvar.T @ (psi ** 2).T)
        predvar = (gvar.T @ (psi ** 2).T) + lsigma2s.exp()

        ypred = self.tx_y(predmean)
        yconfvar = confvar.T * self.ystd**2
        ypredvar = predvar.T * self.ystd**2

        if return_fullcov:
            CH = gvar.sqrt().T[:, :, None] * psi.T[None, :, :]
            CH.transpose_(1, 2)
            yfullpredcov = torch.einsum('nij,jkn->nik', CH,
                                        CH.permute(*torch.arange(CH.ndim - 1, -1, -1)))\
                           + lsigma2s.exp().diag()
            yfullpredcov.transpose_(0, 2)
            yfullpredcov *= self.ystd**2
            return ypred, ypredvar, yconfvar, yfullpredcov

        return ypred, ypredvar, yconfvar

    @torch.no_grad()
    def compute_aux_predictive_quantities(self):
        """
        Compute auxiliary quantities for predictions.
        """
        x = self.x
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()

        D = self.diag_D
        # B := Y @ Sigma^{-1/2} @ Phi
        B = (self.y.T / lsigma2s.exp().sqrt()) @ self.phi

        CinvM = torch.zeros([self.q, self.n])
        Th = torch.zeros([self.q, self.n, self.n])

        for k in range(self.q):
            Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

            Wk, Uk = torch.linalg.eigh(Ck)

            # (I + D_k * C_k)^{-1}
            IpdkCkinv = Uk / (1.0 + D[k] * Wk) @ Uk.T

            CkinvMk = IpdkCkinv @ B.T[k]
            Thk = Uk * ((D[k] * Wk**2) / (Wk**2 + D[k] * Wk**3)).sqrt() @ Uk.T

            CinvM[k] = CkinvMk
            Th[k] = Thk
        self.CinvMs = CinvM
        self.Ths = Th

    @staticmethod
    def init_standard_x(x):
        """
        Standardizes training inputs and collects summary information.
        """
        if x.ndim < 2:
            x = x.unsqueeze(1)
        x_max = x.max(0).values
        x_min = x.min(0).values
        xs = (x - x_min) / (x_max - x_min)

        xnorm = torch.zeros(x.shape[1])
        for j in range(x.shape[1]):
            xdist = (x[:, j].reshape(-1, 1) - x[:, j]).abs()
            xnorm[j] = (xdist[xdist > 0]).mean()
        return xs, x_min, x_max, x.clone(), xnorm

    def standardize_x(self, x0):
        """
        Standardizes new inputs.
        """
        if x0.ndim < 2:
            x0 = x0.unsqueeze(1)
        return (x0 - self.x_min) / (self.x_max - self.x_min)

    def tx_x(self, xs):
        """
        Reverts standardization of inputs.
        """
        return xs * (self.x_max - self.x_min) + self.x_min

    @staticmethod
    def standardize_y(y, robust_mean):
        """
        Standardizes outputs and collects summary information.  Uses median and absolute
        deviation if `robust_mean` is True.  Otherwise, use mean and standard deviation.
        """
        if y.ndim < 2:
            y = y.unsqueeze(0)

        ymean = y.mean(1).unsqueeze(1)

        if robust_mean:
            ycenter = y.median(1).values.unsqueeze(1)
            yspread = (y - ycenter).abs().median(1).values.unsqueeze(1)
        else:
            ycenter = ymean
            yspread = y.std(1).unsqueeze(1)

        ys = (y - ycenter) / yspread

        return ys, ycenter, yspread, y.clone()

    def tx_y(self, ys):
        """
        Reverts output standardization.
        """
        return ys * self.ystd + self.ymean

    def neglpost(self):
        """
        Computes negative log posterior function.
        """
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        x = self.x
        y = self.y

        pc = self.penalty_const

        n = self.n
        q = self.q
        D = self.diag_D
        phi = self.phi
        psi = (phi.T / lsigma2s.exp().sqrt()).T

        nlp = 0
        nlp += n/2 * lsigma2s.sum()
        nlp += 1/2 * ((y.T / lsigma2s.exp().sqrt()) ** 2).sum()

        for k in range(q):
            Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])
            Wk, Uk = torch.linalg.eigh(Ck)

            Qk = Uk / (D[k] + 1/Wk) @ Uk.T   # Qk = inv(dk In + Ckinv) = dkInpCkinv_inv
            Pk = psi.T[k].outer(psi.T[k])

            yQk = y @ Qk
            yPk = y.T @ Pk.T

            nlp += 1/2 * (1 + D[k] * Wk).log().sum()
            nlp -= 1/2 * (yQk * yPk.T).sum()

        # regularization
        nlp += pc['lLmb'] * (lLmb ** 2).sum() + pc['lLmb0'] * (2/n) * (lLmb0 ** 2).sum()
        nlp += -(lnugGPs + 100).log().sum()

        nlp /= n
        return nlp

    def get_param(self):
        """
        Returns the parameters for LCGP instance.
        """
        if self.parameter_clamp_flag:
            lLmb, lLmb0, lsigma2s, lnugGPs = \
                self.parameter_clamp(lLmb=self.lLmb, lLmb0=self.lLmb0,
                                     lsigma2s=self.lsigma2s, lnugs=self.lnugGPs)
        else:
            lLmb, lLmb0, lsigma2s, lnugGPs = \
                self.lLmb, self.lLmb0, self.lsigma2s, self.lnugGPs
        return lLmb, lLmb0, lsigma2s, lnugGPs

    @staticmethod
    def parameter_clamp(lLmb, lLmb0, lsigma2s, lnugs):
        """
        Set soft boundary for parameters.
        """
        d = torch.tensor(lLmb.shape[1],)
        lLmb = (parameter_clamping(lLmb.T,
                                   torch.tensor((-2.5 + 1/2 * torch.log(d), 2.5)))).T
        lLmb0 = parameter_clamping(lLmb0, torch.tensor((-4, 4)))
        lsigma2s = parameter_clamping(lsigma2s, torch.tensor((-12, 1)))
        lnugs = parameter_clamping(lnugs, torch.tensor((-16, -6)))

        return lLmb, lLmb0, lsigma2s, lnugs

    @torch.no_grad()
    def get_param_grad(self):
        """
        Returns parameter gradients.
        """
        grad = []
        for p in filter(lambda p: p.requires_grad, self.parameters()):
            if p.grad is not None:
                view = p.grad.data.view(-1)
                grad.append(view)
        if len(grad) > 0:
            grad = torch.cat(grad, 0)
        return grad
