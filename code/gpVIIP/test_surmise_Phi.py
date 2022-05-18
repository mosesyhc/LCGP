import torch
import numpy as np
from fayans_support import read_only_complete_data


def surmise_baseline(ntrain, ntest, run=None, seed=None, Phi=None):
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
    # torch.seed()
    ftr = f[:, tr_inds]
    thetatr = theta[tr_inds]
    fte = f[:, te_inds]
    thetate = theta[te_inds]

    # SURMISE BLOCK
    if Phi is not None:
        offset = ftr.mean(1)
        scale = np.ones(ftr.shape[0])
        S = np.ones(Phi.shape[1])
        fs = ((ftr.T - offset) / scale).T
        extravar = torch.mean((fs - Phi @ Phi.T @ fs) ** 2, 1) * (scale ** 2)
        standardpcinfo = {'offset': offset.numpy(),
                          'scale': scale,
                          'U': Phi.numpy(),
                          'S': S,
                          'fs': fs.T.numpy(),
                          'extravar': extravar.numpy()}
    from surmise.emulation import emulator
    emu = emulator(x=x0.numpy(), theta=thetatr.numpy(),
                   f=ftr.numpy(), method='PCGPwM',
                   args={'warnings': True,
                         'standardpcinfo': standardpcinfo})

    emupred = emu.predict(x=x0.numpy(), theta=thetate.numpy())
    emumse = ((emupred.mean() - fte.numpy()) ** 2).mean()
    emutrainmse = ((emu.predict().mean() - ftr.numpy())**2).mean()
    print('surmise mse: {:.3f}'.format(emumse))
    print('surmise training mse: {:.3f}'.format(emutrainmse))

    return np.array((run, seed, emumse, emutrainmse))
