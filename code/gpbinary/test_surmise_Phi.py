import torch
import numpy as np
from fayans_support import read_only_complete_data


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
