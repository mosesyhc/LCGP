import torch
from torch import nn
from gp_mvlatent_model import MVlatentGP
from fayans_support import read_data, get_psi, get_empPhi, read_only_complete_data
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device(0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_bestPhi(f):
    U, S, V = torch.linalg.svd(f, full_matrices=False)

    return U


def returnPhi(Phi):
    return Phi


def test_mvlatent():
    f, x0, theta = read_only_complete_data(r'code/data/')

    f = torch.tensor(f)
    x0 = torch.tensor(x0)
    theta = torch.tensor(theta)

    m, n = f.shape

    ntrain = 50
    ntest = 200

    tempind = torch.randperm(n)
    tr_inds = tempind[:ntrain]
    te_inds = tempind[-ntest:]

    ftr = f[:, tr_inds]
    thetatr = theta[tr_inds]
    fte = f[:, te_inds]
    thetate = theta[te_inds]

    # SURMISE BLOCK
    # # if False:
    # from surmise.emulation import emulator
    # emu = emulator(x=x.numpy(), theta=thetatr.numpy(),
    #                f=ftr.numpy(), method='PCGPwM',
    #                args={'warnings': True})
    #
    # emupred = emu.predict(x=x.numpy(), theta=thetate.numpy())
    # emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    # emutrainmse = ((emu.predict().mean() - ftr.numpy())**2).mean()
    # print('surmise mse: {:.3f}'.format(emumse))
    # print('surmise training mse: {:.3f}'.format(emutrainmse))

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


    nepoch_nn = 500
    print('Neural network training:')
    print('{:<5s} {:<12s}'.format('iter', 'train MSE'))
    for epoch in range(nepoch_nn):
        optim_nn.zero_grad()
        l = loss(Phi_as_param, ftr - psi)
        l.backward()
        optim_nn.step()
        if (epoch % 50 - 1) == 0:
            print('{:<5d} {:<12.6f}'.format(epoch, l))
    Phi_match = Phi_as_param().detach()

    if False:
        plt.style.use(['science', 'grid', 'no-latex'])
        plt.figure()
        # plt.hist((Phi0 @ Phi0.T @ (ftr - psi) - (ftr - psi)).reshape(-1).numpy(), bins=30, density=True, label='SVD')
        plt.hist((Phi_match @ torch.linalg.solve(Phi_match.T @ Phi_match,
                  Phi_match.T @ (ftr - psi)) - (ftr - psi)).reshape(-1).numpy(), bins=30,
                 fill=False, density=True, label='matchNN')
        plt.xlabel(r'prediction error')
        plt.ylabel(r'density')
        plt.tight_layout()
        plt.show(block=False)
    print('Reproducing Phi0 error in prediction of F: ',
          torch.mean((Phi_match @ Phi_match.T @ F - F)**2))

    Phi = Phi_match
    G = (Phi.T @ F).T

    # Phi @ G.T \approx Up @ W.T
    print('MSE between Phi @ G.T and (ftr - psi): {:.3f}'.format(torch.mean((Phi @ G.T - (ftr - psi))**2)))

    print('Basis size: ', Phi.shape)

    Lmb = torch.randn(kap, d+1)
    model = MVlatentGP(Lmb=Lmb, G=G, Phi=Phi, lsigma=lsigma, theta=thetatr, f=F, psi=torch.zeros_like(psi))
    model.double()
    model.requires_grad_()

    ftrpred = model(thetatr)
    print('GP training MSE: {:.3f}'.format(torch.mean((F - ftrpred)**2)))
    # optim = torch.optim.LBFGS(model.parameters(), lr=10e-2, line_search_fn='strong_wolfe')
    optim = torch.optim.AdamW(model.parameters(), lr=10e-3)  #, line_search_fn='strong_wolfe')
    nepoch_gp = 500

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
        if epoch % 10 == 0:
            print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format(epoch, lik, mse, trainmse))


if __name__ == '__main__':
    test_mvlatent()
