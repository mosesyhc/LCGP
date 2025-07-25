import numpy as np
import time
from functions import forrester2008
from call_model import LCGPRun
from lcgp import evaluation
import pandas as pd
import pathlib

outputdir = 'submethod-illustration/output/_meanvar/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

rep_n = 1
ns = [250]  #
ntest = 800

submethods = ['elbo', 'proflik', 'full']  # 'full',
for n in ns:
    xtrain = np.linspace(0, 1, n)
    xtest = np.linspace(1 / ntest, 1, ntest)

    ytrue = forrester2008(xtest, noisy=False)
    generating_noises_var = np.array([0.005, 0.1, 0.3]) * ytrue.var(1)

    for i in range(rep_n):
        ytrain = forrester2008(xtrain, noisy=True, noises=generating_noises_var)
        ytest = forrester2008(xtest, noisy=True, noises=generating_noises_var)

        data = {
            'xtrain': xtrain,
            'xtest': xtest,
            'ytrain': ytrain,
            'ytest': ytest,
            'ytrue': ytrue
        }

        for k, model in enumerate([LCGPRun, LCGPRun, LCGPRun]):
            modelrun = model(runno='n{:d}_runno{:d}'.format(n, i), data=data,
                             num_latent=3, submethod=submethods[k], verbose=False)
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
                'submethod': modelrun.submethod,
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
                outputdir + '{:s}_{:s}_{:s}.csv'.format(modelrun.modelname,
                                                        modelrun.submethod,
                                                        modelrun.runno))

            np.savetxt(
                outputdir + '{:s}_{:s}_{:s}_predmean.csv'.format(modelrun.modelname,
                                                                 modelrun.submethod,
                                                                 modelrun.runno),
                predmean
            )
            np.savetxt(
                outputdir + '{:s}_{:s}_{:s}_predvar.csv'.format(modelrun.modelname,
                                                                modelrun.submethod,
                                                                modelrun.runno),
                predvar
            )
            np.savetxt(
                outputdir + '{:s}_{:s}_{:s}_gmean.csv'.format(modelrun.modelname,
                                                                 modelrun.submethod,
                                                                 modelrun.runno),
                modelrun.model.ghat.detach().numpy()
            )
            np.savetxt(
                outputdir + '{:s}_{:s}_{:s}_gvar.csv'.format(modelrun.modelname,
                                                                modelrun.submethod,
                                                                modelrun.runno),
                modelrun.model.gvar.detach().numpy()
            )
