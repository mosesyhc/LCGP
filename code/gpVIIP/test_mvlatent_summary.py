import torch
from torch import nn
from gp_mvlatent_model import MVlatentGP
from fayans_support import read_data, get_psi, get_empPhi, read_only_complete_data
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device(0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np


def get_bestPhi(f):
    U, S, V = torch.linalg.svd(f, full_matrices=False)

    return U


def surmise_baseline(run=None, seed=None):
    f, x0, theta = read_only_complete_data(r'code/data/')

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)

    m, n = f.shape  # nloc, nparam

    ntrain = 50
    ntest = 200

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

    # SURMISE BLOCK
    from surmise.emulation import emulator
    emu = emulator(x=x0.numpy(), theta=thetatr.numpy(),
                   f=ftr.numpy(), method='PCGPwM',
                   args={'warnings': True})

    emupred = emu.predict(x=x0.numpy(), theta=thetate.numpy())
    emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    emutrainmse = ((emu.predict().mean() - ftr.numpy())**2).mean()
    print('surmise mse: {:.3f}'.format(emumse))
    print('surmise training mse: {:.3f}'.format(emutrainmse))

    return np.array((run, seed, emumse, emutrainmse))


def test_mvlatent(run=None, seed=None, nepoch_nn=100, nepoch_gp=200, optim_param_set=None):
    # result storage vectors
    store_Phi_mse = np.zeros((nepoch_nn, 4))
    store_GP_mse = np.zeros((nepoch_gp, 6))

    f, x0, theta = read_only_complete_data(r'code/data/')

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)

    m, n = f.shape  # nloc, nparam

    ntrain = 50
    ntest = 200

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

    psi = ftr.mean(1).unsqueeze(1)
    d = theta.shape[1]
    kap = 20

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
    Phi0 = get_bestPhi(F)
    print('SVD error: ', torch.mean((Phi0 @ Phi0.T @ F - F)**2))

    Phi_as_param = Basis(m, kap, normalize=True)
    optim_nn = torch.optim.Adam(Phi_as_param.parameters(), lr=10e-3)

    print('F shape:', F.shape)

    print('Phi training:')
    print('{:<5s} {:<12s}'.format('iter', 'train MSE'))
    for epoch in range(nepoch_nn):
        optim_nn.zero_grad()
        l = loss(Phi_as_param, ftr - psi)
        l.backward()
        optim_nn.step()

        store_Phi_mse[epoch] = (run, seed, epoch, l.detach().numpy())

        if (epoch % 50 - 1) == 0:
            print('{:<5d} {:<12.6f}'.format(epoch, l))
    Phi_match = Phi_as_param().detach()

    print('Reproducing Phi0 error in prediction of F: ',
          torch.mean((Phi_match @ Phi_match.T @ F - F)**2))

    Phi = Phi_match
    G = (Phi.T @ F).T

    # Phi @ G.T \approx Up @ W.T
    print('MSE between Phi @ G.T and (ftr - psi): {:.3f}'.format(
        torch.mean((Phi @ G.T - F)**2)))
    print('Basis size: ', Phi.shape)

    Lmb = torch.randn(kap, d+1)
    sigma = torch.Tensor(torch.zeros(1))
    model = MVlatentGP(Lmb=Lmb, G=G, Phi=Phi,
                       lsigma=sigma, theta=thetatr,
                       f=F, psi=torch.zeros_like(psi),
                       optim_param_set=optim_param_set)
    model.double()

    # # .requires_grad_() turns all parameters on.
    # model.requires_grad_()
    # print(list(model.parameters()))

    ftrpred = model(thetatr)
    print('GP training MSE: {:.3f}'.format(torch.mean((F - ftrpred)**2)))
    print('Optimizing parameters: ', model.optim_param_set)

    # optim = torch.optim.LBFGS(model.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=10e-3)  #, line_search_fn='strong_wolfe')

    header = ['iter', 'negloglik', 'test mse', 'train mse']
    print('\nGP training:')
    print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
    for epoch in range(nepoch_gp):
        optim.zero_grad()
        lik = model.lik()
        lik.backward()
        optim.step() #lambda: model.lik())

        mse = model.test_mse(thetate, fte - psi)
        trainmse = model.test_mse(thetatr, ftr - psi)

        store_GP_mse[epoch] = (run, seed, epoch, lik.detach().numpy(),
                               mse.detach().numpy(),
                               trainmse.detach().numpy())
        if epoch % 25 == 0:
            print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format(
                epoch, lik, mse, trainmse))

    return store_Phi_mse, store_GP_mse


if __name__ == '__main__':
    nepoch_nn = 100
    nepoch_gp = 300
    nrun = 1
    results = {'surmise': list(),
               'optimTFFT_Phi': list(),
               'optimTFFT_GP': list(),
               'optimTTFT_Phi': list(),
               'optimTTFT_GP': list()}

    for run in range(nrun):
        seed = torch.randint(0, 10000, (1,)).numpy()[0]
        results['surmise'].append(surmise_baseline(run, seed))
        Phi_mse1, GP_mse1 = \
            test_mvlatent(run, seed, nepoch_nn=nepoch_nn,
                          nepoch_gp=nepoch_gp,
                          optim_param_set={
                              'Lmb': True,
                              'G': False,
                              'Phi': False,
                              'lsigma': True
                          })
        results['optimTFFT_Phi'].append(Phi_mse1)
        results['optimTFFT_GP'].append(GP_mse1)

        Phi_mse2, GP_mse2 = \
            test_mvlatent(run, seed, nepoch_nn=nepoch_nn,
                          nepoch_gp=nepoch_gp,
                          optim_param_set={
                              'Lmb': True,
                              'G': True,
                              'Phi': False,
                              'lsigma': True
                          })
        results['optimTTFT_Phi'].append(Phi_mse2)
        results['optimTTFT_GP'].append(GP_mse2)
    # test_mvlatent(run, seed, nepoch_nn=nepoch_nn,
    #               nepoch_gp=nepoch_gp,
    #               optim_param_set={
    #                   'Lmb': True,
    #                   'G': True,
    #                   'Phi': True,
    #                   'lsigma': True
    #               })

    for key, item in results.items():
        results[key] = np.array(item)

    from datetime import datetime
    np.save('testresults_mvlatent_{:s}.npy'.format(datetime.today().strftime('%Y%m%d%H%M')), results)