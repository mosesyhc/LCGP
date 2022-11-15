import numpy as np
from scipy.stats.qmc import LatinHypercube


def get_function(fname):
    if fname == 'borehole':
        from TestingfunctionBorehole import borehole_model as _model, _dict as _dict
    elif fname == 'piston':
        from TestingfunctionPiston import Piston_model as _model, _dict as _dict
    elif fname == 'wingweight':
        from TestingfunctionWingweight import Wingweight_model as _model, _dict as _dict
    elif fname == 'otlcircuit':
        from TestingfunctionOTLcircuit import OTLcircuit_model as _model, _dict as _dict
    else:
        raise ValueError('function name {:s} invalid.'.format(fname))
    return _model, _dict


def gen_data(fname, ntrain, ntest, m, save=True, savedir=None):
    _model, _dict = get_function(fname)

    xdim = _dict['xdim']
    tdim = _dict['thetadim']

    sampler_x = LatinHypercube(d=xdim)
    sampler_t = LatinHypercube(d=tdim)

    x = sampler_x.random(m)
    thetatr = sampler_t.random(ntrain)
    thetate = sampler_t.random(ntest)

    ftr = _model(x=x, theta=thetatr)
    fte = _model(x=x, theta=thetate)

    if save:
        np.savetxt(savedir + r'\x.txt', x)
        np.savetxt(savedir + r'\theta.txt', thetatr)
        np.savetxt(savedir + r'\f.txt', ftr)
        np.savetxt(savedir + r'\testtheta.txt', thetate)
        np.savetxt(savedir + r'\testf.txt', fte)


if __name__ == '__main__':
    for fname in ['borehole', 'piston', 'otlcircuit', 'wingweight']:
        gen_data(fname, 2000, 500, 50, save=True, savedir=r'C:\Users\moses\Desktop\git\VIGP\code\data\{:s}_data'.format(fname))