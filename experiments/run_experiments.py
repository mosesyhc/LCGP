import numpy as np
import time
from experiments.testfunc_wrapper import TestFuncCaller
from call_model import LCGPRun, SVGPRun, OILMMRun, GPPCARun
from lcgp import evaluation
from pyDOE import lhs
import scipy.stats as sps
import pandas as pd
import pathlib

outputdir = r'experiments/output/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

rep_n = 2
output_dims = [8]
ns = [50, 100, 250, 500]
funcs = ['borehole']
ntest = 800

for function in funcs:
    func_caller = TestFuncCaller(function)
    func_meta = func_caller.info
    xdim = func_meta['thetadim']
    locationdim = func_meta['xdim']

    xtest = lhs(xdim, ntest)
    for outputdim in output_dims:
        locations = sps.uniform.rvs(0, 1, (outputdim, locationdim))
        ytrue = func_meta['nofailmodel'](locations, xtest)
        for ntrain in ns:
            generating_noises_var = 0.05 ** ((np.arange(outputdim) + 1) / 2) * np.var(
                ytrue, 1)

            for i in range(rep_n):
                xtrain = lhs(xdim, ntrain)

                # evaluate true models
                ytrain = func_meta['nofailmodel'](locations, xtrain)
                ytest = func_meta['nofailmodel'](locations, xtest)

                # add noise
                ytrain += np.random.normal(np.zeros_like(generating_noises_var),
                                           generating_noises_var,
                                           (ntrain, outputdim)).T
                ytest += np.random.normal(np.zeros_like(generating_noises_var),
                                          generating_noises_var,
                                          (ntest, outputdim)).T

                data = {
                    'xtrain': xtrain,
                    'xtest': xtest,
                    'ytrain': ytrain,
                    'ytest': ytest,
                    'ytrue': ytrue
                }

                robust = [True, False, None, None, None]
                for k, model in enumerate([LCGPRun, LCGPRun, SVGPRun,
                                           OILMMRun]):  # , GPPCARun, LCGPRun, LCGPRun, SVGPRun, OILMMRun
                    modelrun = model(runno='n{:d}_runno{:d}'.format(ntrain, i),
                                     data=data,
                                     num_latent=int(outputdim * 3 / 4),
                                     robust=robust[k])

                    if model == GPPCARun:
                        modelrun.define_model(directory=outputdir.split('/')[0])
                    else:
                        modelrun.define_model()
                    traintime0 = time.time()
                    modelrun.train()
                    traintime1 = time.time()
                    predmean, predvar = modelrun.predict()

                    rmse = evaluation.rmse(ytrue, predmean)
                    nrmse = evaluation.normalized_rmse(ytrue, predmean)
                    pcover, pwidth = evaluation.intervalstats(ytest, predmean, predvar)
                    dss = evaluation.dss(ytrue, predmean, predvar, use_diag=True)

                    result = {
                        'modelname': modelrun.modelname,
                        'runno': modelrun.runno,
                        'function': function,
                        'modelrun': modelrun.runno + '_{:s}_{:d}'.format(function,
                                                                         outputdim),
                        'n': modelrun.n,
                        'traintime': traintime1 - traintime0,
                        'rmse': rmse,
                        'nrmse': nrmse,
                        'pcover': pcover,
                        'pwidth': pwidth,
                        'dss': dss
                    }

                    df = pd.DataFrame.from_dict(result, orient='index').reset_index()
                    df.to_csv(outputdir +
                              '{:s}_{:s}.csv'.format(result['modelname'],
                                                     result['modelrun']))
