import numpy as np
import time
from functions import forrester2008
from lcgp import evaluation
from call_model import LCGPRun
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ns = [200, 400]
ntest = 800

results_rmse = {}
results_fig_path = './results_figure/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)

for n in ns:
    # print(f'Testing number of samples: {n}')
    xtrain = np.linspace(0, 1, n)
    xtest = np.linspace(1 / ntest, 1, ntest)

    ytrue = forrester2008(xtest, noisy=False)
    generating_noises_var = np.array([0.005, 0.1, 0.3]) * ytrue.var(1)

    ytrain = forrester2008(xtrain, noisy=True, noises=generating_noises_var)
    ytest = forrester2008(xtest, noisy=True, noises=generating_noises_var)

    data = {
        'xtrain': xtrain,
        'xtest': xtest,
        'ytrain': ytrain,
        'ytest': ytest,
        'ytrue': ytrue
    }
    modelrun = LCGPRun(runno='n{:d}_runno{:d}'.format(n, 0), data=data,
                       num_latent=ytrain.shape[0]-1)
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

    fig, ax = plt.subplots(3, 1, figsize=(10, 5))
    for i in range(3):
        ax[i].scatter(xtrain, ytrain[i, :], label='Training Data', color='red')
        ax[i].plot(xtest, ytrue[i, :], label='True', color='black')
        ax[i].plot(xtest, predmean[i, :], label='Prediction', color='blue')
        ax[i].fill_between(xtest, predmean[i, :] - 2 * np.sqrt(predvar[i, :]),
                         predmean[i, :] + 2 * np.sqrt(predvar[i, :]), color='blue',
                         alpha=0.3, label='95% CI')

    plt.savefig(results_fig_path + f'lcgp-illustration-n-{n}.png')
    results_rmse[n] = rmse

print(f'RMSE: {results_rmse}')
