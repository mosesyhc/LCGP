import numpy as np
import time
from functions import forrester2008
from call_model import LCGPRun, SVGPRun, OILMMRun, GPPCARun
from lcgp import evaluation
import pandas as pd

outputdir = 'illustration-examples/output/'


ns = [50, 100, 250, 500]
ntest = 800

for n in ns:
    xtrain = np.linspace(0, 1, n)
    xtest = np.linspace(1 / ntest, 1, ntest)

    ytrue = forrester2008(xtest, noisy=False)
    generating_noises_var = np.array([0.005, 0.1, 0.3]) * ytrue.var(1)

    for i in range(5):
        ytrain = forrester2008(xtrain, noisy=True, noises=generating_noises_var)
        ytest = forrester2008(xtest, noisy=True, noises=generating_noises_var)

        data = {
            'xtrain': xtrain,
            'xtest': xtest,
            'ytrain': ytrain,
            'ytest': ytest,
            'ytrue': ytrue
        }
        for model in [LCGPRun, SVGPRun, OILMMRun, GPPCARun]:
            modelrun = model(runno='n{:d}_runno{:d}'.format(n, i), data=data,
                             num_latent=ytrain.shape[0])
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
                'modelrun': modelrun.runno,
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
                outputdir + '{:s}_{:s}.csv'.format(modelrun.modelname, modelrun.runno))
