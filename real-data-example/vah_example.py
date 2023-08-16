import numpy as np
import time
from call_model import LCGPRun, PCSKRun, OILMMRun, SVGPRun
from lcgp import evaluation
from sklearn.model_selection import KFold
import pandas as pd
import pathlib

outputdir = 'real-data-example/output/'
datadir = 'real-data-example/data/'

pathlib.Path(outputdir).mkdir(exist_ok=True)

with open(datadir + '/xlabel.txt', 'r') as f:
    xlabel = f.readlines()
f.close()

xlabel_groups = {}
xlabel_group_counts = {}

for item in xlabel:
    item = item.strip('\n')
    category = item.split('_')[0].strip()
    if not category == 'pT':
        if category in xlabel_groups:
            xlabel_groups[category].append(item)
            xlabel_group_counts[category] += 1
        else:
            xlabel_groups[category] = [item]
            xlabel_group_counts[category] = 1

err_struct = list(xlabel_group_counts.values())

xfull = np.loadtxt(datadir + '/xfull.txt')
yfull = np.loadtxt(datadir + '/yfull.txt')
yfullstd = np.loadtxt(datadir + '/yfullstd.txt').T

num_cv = 5
kfold = KFold(n_splits=num_cv, shuffle=True, random_state=42)
# xtrain = np.loadtxt(datadir + '/xtrain.txt')
# ytrain = np.loadtxt(datadir + '/ytrain.txt')
# xtest = np.loadtxt(datadir + '/xtest.txt')
# ytest = np.loadtxt(datadir + '/ytest.txt')

for run, (train_index, test_index) in enumerate(kfold.split(xfull)):
    data = {
        'xtrain': xfull[train_index],
        'xtest': xfull[test_index],
        'ytrain': yfull[:, train_index],
        'ytest': yfull[:, test_index],
        'ystd': yfullstd[:, train_index]
    }

    robusts = [True, False, None, None, None]
    for k, model in enumerate([LCGPRun, LCGPRun, PCSKRun, OILMMRun, SVGPRun]):
        modelrun = model(runno=str(run), data=data,
                         num_latent=12, robust=robusts[k],
                         err_struct=err_struct)
        modelrun.define_model()
        traintime0 = time.time()
        modelrun.train()
        traintime1 = time.time()
        predmean, predvar = modelrun.predict()

        rmse = evaluation.rmse(data['ytest'], predmean)
        nrmse = evaluation.normalized_rmse(data['ytest'], predmean)
        pcover, pwidth = evaluation.intervalstats(data['ytest'], predmean, predvar)
        dss = evaluation.dss(data['ytest'], predmean, predvar, use_diag=True)

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
            outputdir + '{:s}_{:s}_cv{:s}_lat{:d}.csv'.format(modelrun.modelname,
                                                              result['function'],
                                                              modelrun.runno,
                                                              modelrun.num_latent))
        del modelrun
