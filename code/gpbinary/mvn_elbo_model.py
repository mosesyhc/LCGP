import torch
import torch.nn as nn


class MVN_elbo(nn.Module):
    def __init__(self, mu, v, Phi, lsigma, psi):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.v = nn.Parameter(v)
        self.Phi = Phi
        self.lsigma = nn.Parameter(lsigma)
        self.psi = psi

    def forward(self):
        mu = self.mu
        v = self.v

        psi = self.psi
        Phi = self.Phi

        fpred = psi + (Phi @ mu).unsqueeze(1)

        return fpred

    def elbo(self):
        mu = self.mu
        v = self.v
        psi = self.psi
        Phi = self.Phi
        lsigma = self.lsigma

        m, kap = Phi.shape

        s2inv = torch.exp(-2 * lsigma)

        Rinv =  torch.zeros((m+kap, m+kap))
        Rinv[:kap, :kap] = torch.diag(1/v) + s2inv * Phi.T @ Phi
        Rinv[:kap, kap:(m+kap)] = -s2inv * Phi.T
        Rinv[kap:(m+kap), :kap] = -s2inv * Phi
        Rinv[kap:(m+kap), kap:(m+kap)] = s2inv * torch.eye(m)

        Z = torch.zeros((m+kap))
        Z[:kap] = mu
        Z[kap:(m+kap)] = psi.squeeze() + Phi @ mu

        negelbo = 1/2 * Z.T @ Rinv @ Z - 1/2 * torch.sum(torch.log(v))

        return negelbo

    def test_mse(self, f0):
        mu = self.mu
        v = self.v
        psi = self.psi
        Phi = self.Phi
        lsigma = self.lsigma

        fpred = psi + (Phi @ mu).unsqueeze(1)

        return ((fpred - f0) ** 2).mean()
