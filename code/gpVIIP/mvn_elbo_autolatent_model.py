import torch
import torch.nn as nn
import torch.jit as jit
from matern_covmat import covmat
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
            ghat[k], _ = pred_gp(llmb=lLmb[k], theta=theta, thetanew=theta0, g=M[k])

        fhat = Phi @ ghat
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

        m = self.m
        n = self.n
        kap = self.kap

        M = self.M
        V = self.V

        negelbo = 0
        for k in range(kap):
            negloggp_k, _ = negloglik_gp(llmb=lLmb[k], theta=theta, g=M[k])
            negelbo += negloggp_k

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

        n = self.n
        Phi = self.Phi
        F = self.F
        sigma2 = torch.exp(lsigma2)

        M = torch.zeros(self.kap, self.n)
        V = torch.zeros(self.kap, self.n)
        for k in range(kap):
            C_k = covmat(theta, theta, lLmb[k])
            W_k, U_k = torch.linalg.eigh(C_k)
            Winv_k = 1 / W_k
            Mk = torch.linalg.solve(torch.eye(n) + sigma2 * U_k * Winv_k @ U_k.T, Phi[:, k] @ F)
            M[k] = Mk
            V[k] = 1 / (1 / sigma2 + torch.diag((U_k * Winv_k) @ U_k.T))
        self.M = M
        self.V = V

    def predictmean(self, theta0):
        self.compute_MV()
        fhat = self.forward(theta0)
        return(fhat)

    def predictcov(self, theta0):
        self.compute_MV()


    def predictvar(self, theta0):
        self.compute_MV()
        return

    def test_mse(self, theta0, f0):
        with torch.no_grad():
            fhat = self.predictmean(theta0)
            return ((fhat - f0) ** 2).mean()

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

    @staticmethod
    def parameter_clamp(lLmb, lsigma2):
        # clamping
        lLmb = (parameter_clamping(lLmb.T, torch.tensor((-2.5, 2.5)))).T
        lsigma2 = parameter_clamping(lsigma2, torch.tensor((-12, -1)))

        return lLmb, lsigma2
