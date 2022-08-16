import torch
from torch import nn
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp
from fayans_support import read_only_complete_data, get_empPhi

torch.set_default_dtype(torch.double)
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)
from test_mvn_elbo import test_mvn_elbo
from test_surmise_Phi import surmise_baseline

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def test_mvn_elbo_autolatent_sp(ntrain, ntest, kap, run=None, seed=None, nepoch_nn=100, nepoch_elbo=400):
    # result storage vectors
    store_Phi_mse = np.zeros((nepoch_nn, 4))
    store_elbo_mse = np.zeros((nepoch_elbo, 6))

    f, x0, theta = read_only_complete_data(r'code/data/fayans_data/')

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)

    m, n = f.shape  # nloc, nparam
    d = theta.shape[1]

    if seed is not None:
        torch.manual_seed(seed)
    tempind = torch.randperm(n)
    tr_inds = tempind[:ntrain]
    te_inds = tempind[-ntest:]
    # torch.seed()
    ftr = f[:, tr_inds]
    thetatr = theta[tr_inds]
    fte = f[:, te_inds]
    thetate = theta[te_inds]

    # choose inducing points
    ni = 20
    kmeans_theta = KMeans(n_clusters=ni).fit(thetatr)
    thetai = torch.Tensor(kmeans_theta.cluster_centers_)
    D = pairwise_distances(thetatr, thetai)
    overlap_inds = torch.where(torch.tensor(D) == 0)[1]
    print(overlap_inds)
    # for i in overlap_inds:
    #     thetai[i] += torch.normal(torch.zeros(d), 0.1 * thetai.std(0))

    psi = ftr.mean(1).unsqueeze(1)

    x = torch.column_stack((x0[:, 0], x0[:, 1],
                            *[x0[:, 2] == k for k in torch.unique(x0[:, 2])]))
    F = ftr - psi

    from basis_nn_model import Basis
    def loss(class_Phi, F):
        Phi = class_Phi()
        if class_Phi.normalize:
            Fhat = Phi @ Phi.T @ F
        else:
            Fhat = Phi @ torch.linalg.solve(Phi.T @ Phi, Phi.T) @ F
        mse = nn.MSELoss(reduction='mean')
        return mse(Fhat, F)

    # # get_bestPhi takes F and gives the SVD U
    # Phi_as_param = Basis(m, kap, normalize=True)
    # optim_nn = torch.optim.Adam(Phi_as_param.parameters(), lr=10e-3)
    #
    # print('F shape:', F.shape)
    #
    # print('Neural network training:')
    # print('{:<5s} {:<12s}'.format('iter', 'train MSE'))
    # for epoch in range(nepoch_nn):
    #     optim_nn.zero_grad()
    #     l = loss(Phi_as_param, F)
    #     l.backward()
    #     optim_nn.step()
    #
    #     store_Phi_mse[epoch] = (run, seed, epoch, l.detach().numpy())
    #     if (epoch % 50 - 1) == 0:
    #         print('{:<5d} {:<12.6f}'.format(epoch, l))
    # Phi_match = Phi_as_param().detach()

    Phi_match, _, _ = torch.linalg.svd(F, full_matrices=False)
    Phi_match = Phi_match[:, :kap]

    mse_Phi = torch.mean((Phi_match @ Phi_match.T @ F - F) ** 2)
    print('Reproducing Phi0 error in prediction of F: ', mse_Phi)

    # Pick random inducing points from data for now
    # thetai = theta[torch.randint(0, theta.shape[0], size=(30,))]

    Phi = Phi_match
    print('Basis size: ', Phi.shape)
    # np.savetxt('Phi_seed0.txt', Phi.numpy())
    # Phi = torch.tensor(np.loadtxt('Phi_seed0.txt'))

    lmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                       torch.log(torch.std(theta, 0)))
    lmb = torch.cat((lmb, torch.Tensor([0])))
    Lmb = lmb.repeat(kap, 1)
    Lmb[:, -1] = torch.log(torch.var(Phi.T @ (F - psi), 1))
    lsigma2 = torch.Tensor(torch.log(mse_Phi))
    model = MVN_elbo_autolatent_sp(lLmb=Lmb, initlLmb=True,
                                   lsigma2=lsigma2, initlsigma2=True,
                                   # psi=torch.zeros_like(psi),
                                   Phi=Phi, F=F, theta=thetatr, thetai=thetai)
    model.double()

    from matern_covmat import cormat
    from mvn_elbo_autolatent_sp_model import cov_sp

    # C = covmat(theta[:m], theta[:m], Lmb[0])
    # C_inv = torch.linalg.inv(C)
    # logdet_C = torch.linalg.slogdet(C).logabsdet
    # C_sp, C_sp_inv, logdet_C_sp = cov_sp(theta[:m], theta[100:125], Lmb[0])
    #
    # print(C.shape)
    # print(((C - C_sp)**2).sum())
    # print(((C_inv - C_sp_inv)**2).sum())
    # print(logdet_C, logdet_C_sp)

    # # .requires_grad_() turns all parameters on.
    # model.requires_grad_()
    # print(list(model.parameters()))

    # ftrpred = model(thetatr)
    # print('ELBO model training MSE: {:.3f}'.format(torch.mean((F - ftrpred) ** 2)))
    # optim = torch.optim.LBFGS(model.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=10e-3)  # , line_search_fn='strong_wolfe')
    header = ['iter', 'neg elbo', 'test mse', 'train mse']
    print('\nELBO training:')
    print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(nepoch_elbo):
        optim.zero_grad()
        negelbo = model.negelbo()
        negelbo.backward(retain_graph=True)
        optim.step()  # lambda: model.lik())

        mse = model.test_mse(thetate, fte - psi)
        trainmse = model.test_mse(thetatr, F)

        store_elbo_mse[epoch] = (run, seed, epoch, negelbo.detach().numpy(),
                                 mse.detach().numpy(),
                                 trainmse.detach().numpy())

        # if epoch % 10 == 0:
        print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format
              (epoch, negelbo, mse, trainmse))

    return store_Phi_mse, store_elbo_mse, Phi


if __name__ == '__main__':
    nepoch_nn = 100
    nepoch_elbo = 300
    ntrain = 50
    ntest = 50
    kap = 20
    nrun = 3
    results = {
        'surmise': list(),
        'optim_Phi': list(),
        'optim_elbo': list(),
        'optim_Phi_autolatent': list(),
        'optim_elbo_autolatent': list()
    }

    for run in range(nrun):
        seed = torch.randint(0, 10000, (1,)).numpy()[0]

        Phi_mse1, elbo_mse1, Phi = \
            test_mvn_elbo_autolatent_sp(
                ntrain=ntrain, ntest=ntest, kap=kap,
                run=run, seed=seed, nepoch_nn=nepoch_nn,
                nepoch_elbo=nepoch_elbo)
        results['optim_Phi_autolatent'].append(Phi_mse1)
        results['optim_elbo_autolatent'].append(elbo_mse1)

        results['surmise'].append(surmise_baseline(
            ntrain=ntrain, ntest=ntest,
            run=run, seed=seed, Phi=Phi))

        Phi_mse2, elbo_mse2 = \
            test_mvn_elbo(
                ntrain=ntrain, ntest=ntest, kap=kap,
                run=run, seed=seed, nepoch_nn=nepoch_nn,
                nepoch_elbo=nepoch_elbo, optimMu=False, optimV=False)
        results['optim_Phi'].append(Phi_mse2)
        results['optim_elbo'].append(elbo_mse2)

    for key, item in results.items():
        results[key] = np.array(item)
    #
    # dir = r'C:\Users\moses\Desktop\git\binary-hd-emulator\code\test_results\elbo_20220308'
    # from datetime import datetime
    #
    # np.save(dir + r'\testresults_mvnelbo_{:s}.npy'.format(datetime.today().strftime('%Y%m%d%H%M')), results)
