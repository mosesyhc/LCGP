import numpy as np
from scipy.stats.qmc import LatinHypercube
from TestingfunctionBorehole import borehole_model, _dict as borehole_dict


def gen_data(ntrain, ntest, m, save=True, savedir=None):
    xdim = borehole_dict['xdim']
    tdim = borehole_dict['thetadim']

    sampler_x = LatinHypercube(d=xdim)
    sampler_t = LatinHypercube(d=tdim)

    x = sampler_x.random(m)
    thetatr = sampler_t.random(ntrain)
    thetate = sampler_t.random(ntest)

    ftr = borehole_model(x=x, theta=thetatr)
    fte = borehole_model(x=x, theta=thetate)

    if save:
        np.savetxt(savedir + r'\x.txt', x)
        np.savetxt(savedir + r'\theta.txt', thetatr)
        np.savetxt(savedir + r'\f.txt', ftr)
        np.savetxt(savedir + r'\testtheta.txt', thetate)
        np.savetxt(savedir + r'\testf.txt', fte)


if __name__ == '__main__':
    gen_data(2000, 500, 100, save=True, savedir=r'C:\Users\cmyh\Documents\git\binary-hd-emulator\code\data\borehole_m100_data')