import torch
import torch.distributions.normal as Normal
from matern_covmat import covmat
norm = Normal.Normal(loc=0, scale=1)


# named inputs in function calls
def negloglik_mvbinary(lmb, G, theta, y, psi, Phi):
    # hyperparameter organization (size 2d + 2 + 1 + kap*n), kap = 2:
    # hyp = (lambda_1, lambda_2, sigma, G_11, G_21, ..., G_n1, G_12, ..., Gn2)
    # (lambda_k1, ..., lambda_kd) are the lengthscales for theta, k = 1, 2
    # lambda_k(d+1) is the scale for GP, k = 1, 2
    # sigma is the noise parameter in the indicator function

    kap = Phi.shape[1]

    nll = negloglik_link(G, y, psi, Phi)
    for k in range(kap):
        nll += negloglik_gp(lmb[:, k], theta, G[:, k])
    return nll


def negloglik_mvlatent(Lmb, G, theta, f, psi, Phi):
    kap = Phi.shape[1]

    D = f - (psi + Phi @ G.T)
    nll = 1/2 * (D.T @ D).sum()
    for k in range(kap):
        nll += negloglik_gp(lmb=Lmb[k], theta=theta, g=G[:, k])
    return nll


def negloglik_link(G, y, psi, Phi):
    z = (psi + Phi @ G.T)
    F = norm.cdf(z)
    ypos = y > 0.5
    negloglik = -(torch.log(F[ypos])).sum()  # ones
    negloglik -= (torch.log((1 - F[~ypos]))).sum()  # zeros

    return negloglik


def negloglik_gp(lmb, theta, g, lmbregmean=0, lmbregstd=1):
    R = covmat(theta, theta, lmb)

    W, V = torch.linalg.eigh(R)
    Vh = V / torch.sqrt(W)
    fcenter = Vh.T @ g
    n = g.shape[0]

    sig2hat = (n * torch.mean(fcenter ** 2) + 1) / (n + 1)
    negloglik = 1/2 * torch.sum(torch.log(W)) + n/2 * torch.log(sig2hat)
    negloglik += 1/2 * torch.sum(((lmb - lmbregmean + 10e-8) / lmbregstd)**2)


    # term1 = 1/2 * torch.sum(torch.log(W))
    # term2 = n/2 * torch.log(sig2hat)
    # term3 = 1/2 * torch.sum(((lmb - lmbregmean + 10e-8) / lmbregstd)**2)
    # print('{:<6.3f} {:<6.3f} {:<6.3f} {:<6.3f}'.format(negloglik, term1, term2, term3))

    return negloglik


# # delete this
# def negloglikgrad(hyp, theta, y, psi, Phi):
#     dnegloglik = torch.zeros(hyp.shape[0])
#
#     d = theta.shape[1]
#     m, n = y.shape
#     kap = Phi.shape[1]
#
#     G = hyp[-(kap*n):]
#     G.resize_(kap, n).transpose_(0, 1)  # G is n x kap
#     lmb1 = hyp[:(d+1)]
#     lmb2 = hyp[(d+1):(2*d+2)]
#     sigma = hyp[2*d+2]
#
#     dnegloglik[:(d+1)], R1inv = negloglikgrad_gp(lmb1, theta, G[:, 0])
#     dnegloglik[(d+1):(2*d+2)], R2inv = negloglikgrad_gp(lmb2, theta, G[:, 1])
#     dnegloglik[2*d+2:] = negloglikgrad_link(sigma, G, y, psi, Phi)
#     dnegloglik[-(2*n):-n] += R1inv @ G[:, 0]
#     dnegloglik[-n:] += R2inv @ G[:, 1]
#     return dnegloglik


#
# def negloglikgrad_link(sigma, G, y, psi, Phi):
#     # gradient is of size (2n + 1)
#     # first entry is over sigma, the remaining are over G
#     m, n = y.shape
#     kap = Phi.shape[1]
#
#     z = (psi + Phi @ G.T) / sigma
#     F = norm.cdf(z)
#     f = torch.exp(norm.log_prob(z))
#     ypos = y > 0.5
#
#     Phi1mat = Phi[:, 0].repeat(n).resize_(n, m).transpose(1, 0)
#     Phi2mat = Phi[:, 1].repeat(n).resize_(n, m).transpose(1, 0)
#
#     dnegloglik = torch.zeros(kap*n + 1)
#     dnegloglik[0] = (z[ypos] / sigma * f[ypos] / F[ypos]).sum()  # ones
#     dnegloglik[0] -= ((z[~ypos] / sigma) * f[~ypos] / (1 - F[~ypos])).sum()  # zeros
#     dnegloglik[1:(n+1)] = -(Phi1mat / sigma * f / F * ypos).sum(0) + (Phi1mat / sigma * f / (1 - F) * ~ypos).sum(0)
#     dnegloglik[(n+1):] = -(Phi2mat / sigma * f / F * ypos).sum(0) + (Phi2mat / sigma * f / (1 - F) * ~ypos).sum(0)
#     return dnegloglik

#
#
# def negloglikgrad_gp(lmb, theta, g, lmbregmean=None, lmbregstd=None):
#     R, dR = covmat(theta, theta, lmb, return_gradhyp=True)
#
#     W, V = torch.linalg.eigh(R)
#     Vh = V / torch.sqrt(torch.abs(W))
#     fcenter = (Vh.T @ g).unsqueeze(1)
#     n = g.shape[0]
#
#     sig2hat = (n * torch.mean(fcenter ** 2) + 1) / (n + 1)
#     outerprod = Vh @ fcenter @ fcenter.T @ Vh.T
#     Rinv = Vh @ Vh.T
#
#     dnegloglik = torch.zeros(dR.shape[2])
#     for j in range(dR.shape[2]):
#         dsig2hat = - torch.sum(outerprod * dR[:, :, j]) / (n + 1)
#         dnegloglik[j] += 1/2 * n * dsig2hat / sig2hat
#         dnegloglik[j] += 1/2 * torch.sum(Rinv * dR[:, :, j])
#
#     # dnegloglik += (10 ** (-8) + lmb - lmbregmean) / (lmbregstd ** 2)
#     return dnegloglik, Rinv
