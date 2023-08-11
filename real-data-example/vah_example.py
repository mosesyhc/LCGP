import numpy as np
import time
from call_model import LCGPRun, SVGPRun, OILMMRun
from lcgp import evaluation
import pandas as pd
import pathlib

outputdir = 'real-data-example/output/'
datadir = 'real-data-example/data/'

pathlib.Path(outputdir).mkdir(exist_ok=True)

xtrain = np.loadtxt(datadir + '/xtrain.txt')
ytrain = np.loadtxt(datadir + '/ytrain.txt')
xtest = np.loadtxt(datadir + '/xtest.txt')
ytest = np.loadtxt(datadir + '/ytest.txt')

data = {
    'xtrain': xtrain,
    'xtest': xtest,
    'ytrain': ytrain,
    'ytest': ytest
}

robusts = [False, None, None]
for model in [LCGPRun, SVGPRun, OILMMRun]:
    modelrun = model(runno='', data=data,
                     num_latent=25, robust=robusts[0], )
    modelrun.define_model()
    traintime0 = time.time()
    modelrun.train()
    traintime1 = time.time()
    predmean, predvar = modelrun.predict()

    rmse = evaluation.rmse(ytest, predmean)
    nrmse = evaluation.normalized_rmse(ytest, predmean)
    pcover, pwidth = evaluation.intervalstats(ytest, predmean, predvar)
    dss = evaluation.dss(ytest, predmean, predvar, use_diag=True)

    result = {
        'modelname': modelrun.modelname,
        'modelrun': modelrun.runno,
        'function': 'VAH',
        'n': modelrun.n,
        'traintime': traintime1 - traintime0,
        'rmse': rmse,
        'nrmse': nrmse,
        'pcover': pcover,
        'pwidth': pwidth,
        'dss': dss
    }

    df = pd.DataFrame.from_dict(result, orient='index').reset_index()
    df.to_csv(
        outputdir + '{:s}_{:s}.csv'.format(modelrun.modelname, result['function']))
    break