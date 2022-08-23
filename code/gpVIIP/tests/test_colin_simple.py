import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from hyperparameter_tuning import parameter_clamping
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp
from surmise.emulation import emulator
from optim_elbo import optim_elbo, optim_elbo_lbfgs

import matplotlib.pyplot as plt


res_struct = dict.fromkeys(['method', 'rep',
                            'n', 'p', 'seed',
                            'timeconstruct', 'Phi_mse',
                            'trainrmse', 'testrmse',
                            'optim_elbo_iter', 'optim_elbo_flag', 'optim_elbo_lr',
                            'precision'])


def build_surmise(ftr, thetatr, Phi):
    from surmise.emulation import emulator
    emu = emulator(x=np.arange(ftr.shape[0]),
                   theta=thetatr.numpy(),
                   f=ftr.numpy(), method='PCGPwM',
                   args={'warnings': True,
                         'standardpcinfo': {'U': Phi.numpy()}})

    return emu


def rmse_w_surmise(emu, fte, thetate):
    emupred = emu.predictmean(x=np.arange(ftr.shape[0]), theta=thetate.numpy())
    emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    return np.sqrt(emumse)

def dss_surmise(emu, fte, thetate, use_diag=True):
    emupred = emu.predict(x=x, theta=thetate)
    predmean = emupred.mean()
    predcov = np.transpose(emupred.covx(), (0, 2, 1))

    n0 = thetate.shape[0]
    def __dss_single(f, mu, Sigma):  # Dawid-Sebastani score
        r = f - mu
        sign, logdet = np.linalg.slogdet(Sigma)

        score_single = sign*logdet + r @ np.linalg.solve(Sigma, r)
        print(sign*logdet, r @  np.linalg.solve(Sigma, r))
        return score_single

    def __dss_single_diag(f, mu, Sigma):
        r = f - mu
        diagV = np.diag(Sigma)
        score_single = np.log(diagV).sum() + (r * r / diagV).sum()
        print('mse {:.6f}'.format((r**2).mean()))
        print('diag cov mean: {:.6f}'.format(diagV.mean()))
        print('logdet: {:.6f}, quadratic: {:.6f}'.format(np.log(diagV).sum(), (r * r / diagV).sum()))
        return score_single

    __score_single = __dss_single
    if use_diag:
        __score_single = __dss_single_diag

    score = 0
    for i in range(n0):
        score += __score_single(fte[:, i], predmean[:, i], predcov[:, :, i])
    score /= n0
    return score


if __name__ == '__main__':
    data_dir = r'code/data/colin_data/'
    theta = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_input.csv')
    f = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_output.csv')
    theta = torch.tensor(theta.iloc[:, 1:].to_numpy())
    f = torch.tensor(f.iloc[:, 1:].to_numpy()).T

    # f = ((f.T - f.min(1).values) / (f.max(1).values - f.min(1).values)).T

    # arbitrary x
    m, n_all = f.shape
    x = np.arange(m)

    ntr = 300
    indtr = torch.randperm(n_all)[:ntr]
    indte = np.setdiff1d(np.arange(n_all), indtr)[:105]
    ftr = f[:, indtr]
    thetatr = theta[indtr]
    fte = f[:, indte]
    thetate = theta[indte]

    ftr = ((ftr.T - ftr.mean(1)) / ftr.std(1)).T
    fte = ((fte.T - ftr.mean(1)) / ftr.std(1)).T

    Phi, S, _ = torch.linalg.svd(ftr, full_matrices=False)
    v = (S**2).cumsum(0)/(S**2).sum()

    kap = 1  # torch.argwhere(v > 0.995)[0] + 1

    Phi = Phi[:, :kap]
    Phi_mse = ((ftr - Phi @ Phi.T @ ftr)**2).mean()
    print('recovery mse: {:.3E}'.format(Phi_mse))

    model = MVN_elbo_autolatent(Phi=Phi, F=ftr, theta=thetatr, clamping=True)

    print('train mse: {:.3E}'.format(model.test_mse(theta0=thetatr, f0=ftr)))

    model.compute_MV()

    model, niter, flag = optim_elbo_lbfgs(model, maxiter=500, lr=5e-1, gtol=1e-2, thetate=thetate, fte=fte)

    print(model.lLmb)
    print(model.lLmb.grad)
    print(model.lsigma2)
    print(model.lsigma2.grad)




    #
    # print(model.test_rmse(theta0=thetatr, f0=ftr))
    # print('negelbo at optimizer: {:.5f}, sigma2 = {:.3f}'.format(model.negelbo(), parameter_clamping(model.lsigma2, torch.tensor((-12, -1))).exp()))
    # print('grad lsigma2: {:.3f}'.format(model.lsigma2.grad))
    # print('\n')
    # print('negelbo with sigma2 = recovery mse: {:.5f}, sigma2 = {:.3f}'.format(model.negelbo(
    #     lsigma2=torch.log(Phi_mse)), Phi_mse))
    #
    # # print(model.test_rmse(theta0=thetate, f0=fte))
    # #
    # # dss_model = model.dss(theta0=thetate, f0=fte, use_diag=True)
    # # # print('MVGP DS Score: {:.3f}'.format(dss_model_full))
    # chi2 = model.chi2mean(thetate, fte)
    # print(chi2.mean())
    # #
    # predmean = model.predictmean(thetatr)
    # fhat, ghat = model(thetate)
    #
    # predcov, predcov_g = model.predictcov(thetate)
    # #
    # # model._MVN_elbo_autolatent__single_chi2mean(f=fte[:, 0], mu=predmean[:, 0], Sigma=predcov[:, :, 0])
    # # print()
    # #
    # emu = emulator(f=ftr.numpy(),
    #                theta=thetatr.numpy(),
    #                x=x, method='PCGPwM')
    # emupred = emu.predict(x=x, theta=thetate.numpy())
    # emumean = emupred.mean()
    # emuvar = emupred.var()
    #
    # emuchi2 = (emumean ** 2 / emuvar)
    # # emurmse = np.sqrt(((emupred.mean() - fte.numpy())**2).mean())
    #
    # # print('surmise rmse: {:.3f}'.format(emurmse))
    # #
    # # print('DS Score: {:.3f}'.format(dss_surmise(emu, fte[:, :1].numpy(), thetate[:1].numpy(), use_diag=True)))
