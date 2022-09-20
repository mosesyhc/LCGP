import torch
from torch import nn
from mvn_elbo_model import MVN_elbo
from fayans_support import read_only_complete_data

torch.set_default_dtype(torch.double)
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)
from test_surmise_Phi import surmise_baseline


def test_mvn_elbo(ntrain, ntest, kap, run=None, seed=None, nepoch_nn=100, nepoch_elbo=400,
                  optimMu=True, optimV=True):
    # result storage vectors
    store_Phi_mse = np.zeros((nepoch_nn, 4))
    store_elbo_mse = np.zeros((nepoch_elbo, 6))

    f, x0, theta = read_only_complete_data(r'code/data/')

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)

    m, n = f.shape  # nloc, nparam

    if seed is not None:
        torch.manual_seed(seed)
    tempind = torch.randperm(n)
    tr_inds = tempind[:ntrain]
    te_inds = tempind[-ntest:]
    torch.seed()
    ftr = f[:, tr_inds]
    thetatr = theta[tr_inds]
    fte = f[:, te_inds]
    thetate = theta[te_inds]

    psi = ftr.mean(1).unsqueeze(1)
    d = theta.shape[1]

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

    # get_bestPhi takes F and gives the SVD U
    Phi_as_param = Basis(m, kap, normalize=True)
    optim_nn = torch.optim.Adam(Phi_as_param.parameters(), lr=10e-3)

    print('F shape:', F.shape)

    print('Neural network training:')
    print('{:<5s} {:<12s}'.format('iter', 'train MSE'))
    for epoch in range(nepoch_nn):
        optim_nn.zero_grad()
        l = loss(Phi_as_param, F)
        l.backward()
        optim_nn.step()

        store_Phi_mse[epoch] = (run, seed, epoch, l.detach().numpy())
        if (epoch % 50 - 1) == 0:
            print('{:<5d} {:<12.6f}'.format(epoch, l))
    Phi_match = Phi_as_param().detach()

    print('Reproducing Phi0 error in prediction of F: ',
          torch.mean((Phi_match @ Phi_match.T @ F - F) ** 2))

    Phi = Phi_match
    print('Basis size: ', Phi.shape)
    # np.savetxt('Phi_seed0.txt', Phi.numpy())

    # Phi = torch.tensor(np.loadtxt('Phi_seed0.txt'))

    Lmb = torch.zeros(kap, d + 1)
    Mu = Phi.T @ F
    V = torch.ones((kap, ntrain)) * 0.0001
    lsigma2 = torch.Tensor([0])
    model = MVN_elbo(Mu=Mu, V=V, Lmb=Lmb,
                     lsigma2=lsigma2, psi=torch.zeros_like(psi),
                     Phi=Phi, F=F, theta=thetatr, initLmb=True,
                     optimMu=optimMu, optimV=optimV)
    model.double()

    # # .requires_grad_() turns all parameters on.
    # model.requires_grad_()
    # print(list(model.parameters()))

    ftrpred = model(thetatr)
    print('ELBO model training MSE: {:.3f}'.format(torch.mean((F - ftrpred) ** 2)))
    # optim = torch.optim.LBFGS(model.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=2 * 10e-3)  # , line_search_fn='strong_wolfe')
    header = ['iter', 'neg elbo', 'test mse', 'train mse']
    print('\nELBO training:')
    print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(nepoch_elbo):
        optim.zero_grad()
        negelbo = model.negelbo()
        negelbo.backward()
        optim.step()  # lambda: model.lik())

        mse = model.test_mse(thetate, fte - psi)
        trainmse = model.test_mse(thetatr, F)

        store_elbo_mse[epoch] = (run, seed, epoch, negelbo.detach().numpy(),
                                 mse.detach().numpy(),
                                 trainmse.detach().numpy())

        if epoch % 10 == 0:
            print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format
                  (epoch, negelbo, mse, trainmse))

    return store_Phi_mse, store_elbo_mse


if __name__ == '__main__':
    nepoch_nn = 100
    nepoch_elbo = 200
    ntrain = 25
    ntest = 100
    kap = 6
    nrun = 1
    results = {
        'surmise': list(),
        'optim_Phi': list(),
        'optim_elbo': list(),
        'optimMu_Phi': list(),
        'optimMu_elbo': list(),
        'optimMuV_Phi': list(),
        'optimMuV_elbo': list()
    }

    for run in range(nrun):
        seed = torch.randint(0, 10000, (1,)).numpy()[0]
        results['surmise'].append(surmise_baseline(
            ntrain=ntrain, ntest=ntest,
            run=run, seed=seed))

        Phi_mse1, elbo_mse1 = \
            test_mvn_elbo(
                ntrain=ntrain, ntest=ntest, kap=kap,
                run=run, seed=seed, nepoch_nn=nepoch_nn,
                nepoch_elbo=nepoch_elbo,
                optimMu=False,
                optimV=False)
        results['optim_Phi'].append(Phi_mse1)
        results['optim_elbo'].append(elbo_mse1)

        Phi_mse2, elbo_mse2 = \
            test_mvn_elbo(
                ntrain=ntrain, ntest=ntest, kap=kap,
                run=run, seed=seed, nepoch_nn=nepoch_nn,
                nepoch_elbo=nepoch_elbo,
                optimMu=True,
                optimV=False)
        results['optimMu_Phi'].append(Phi_mse2)
        results['optimMu_elbo'].append(elbo_mse2)

        Phi_mse3, elbo_mse3 = \
            test_mvn_elbo(
                ntrain=ntrain, ntest=ntest, kap=kap,
                run=run, seed=seed, nepoch_nn=nepoch_nn,
                nepoch_elbo=nepoch_elbo,
                optimMu=True,
                optimV=True)
        results['optimMuV_Phi'].append(Phi_mse3)
        results['optimMuV_elbo'].append(elbo_mse3)

    for key, item in results.items():
        results[key] = np.array(item)

    dir = r'C:\Users\moses\Desktop\git\binary-hd-emulator\code\test_results\elbo_20220302'
    from datetime import datetime

    np.save(dir + r'\testresults_mvnelbo_{:s}.npy'.format(datetime.today().strftime('%Y%m%d%H%M')), results)