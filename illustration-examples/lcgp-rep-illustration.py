import numpy as np
# import time
from lcgp import LCGP
from lcgp import evaluation
# import pandas as pd
import pathlib


def forrester2008(x, noisy=True, noises=(0.01, 0.1, 0.25)):
    x = np.expand_dims(x, 1) if x.ndim < 2 else x

    y1 = (6*x - 2)**2 * np.sin(12*x - 4)

    def forrester1d(y0, x0, a, b, c):
        return a*y0 + b*(x0 - 0.5) - c
    y2 = forrester1d(y1, x, 0.5, 5, -5)
    y3 = forrester1d(y1, x, -0.8, -5, 4)
    if noisy:
        e1 = np.random.normal(0, np.sqrt(noises[0]), x.shape)
        e2 = np.random.normal(0, np.sqrt(noises[1]), x.shape)
        e3 = np.random.normal(0, np.sqrt(noises[2]), x.shape)
        y1 += e1
        y2 += e2
        y3 += e3
    y = np.row_stack((y1.T, y2.T, y3.T))

    return y


outputdir = r'illustration-examples/rep/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

n = 3
ntest = 1000

xtrain_uniq = np.linspace(0, 1, n)
xtest = np.linspace(1 / ntest, 1, ntest)

reps = np.array((1, 2, 3))
xtrain = np.repeat(xtrain_uniq, reps)

U = np.zeros((sum(reps), len(reps)))
for i in range(len(reps)):
    U[sum(reps[:i]):sum(reps[:(i+1)]), i] = 1

ytrue = forrester2008(xtest, noisy=False)
generating_noises_var = np.array([0.005, 0.1, 0.3]) * ytrue.var(1)

ytrain = forrester2008(xtrain, noisy=True, noises=generating_noises_var)
ytest = forrester2008(xtest, noisy=True, noises=generating_noises_var)

data = {
    'xtrain': xtrain,
    'xtest': xtest,
    'ytrain': ytrain,
    'ytest': ytest,
    'ytrue': ytrue,
    'noisevars': generating_noises_var
}

model = LCGP(y=ytrain,
             x=xtrain,
             )

model.fit()
predmean, predvar = model.predict(xtest)

rmse = evaluation.rmse(ytrue, predmean)
nrmse = evaluation.normalized_rmse(ytrue, predmean)
pcover, pwidth = evaluation.intervalstats(ytest, predmean, predvar)
dss = evaluation.dss(ytrue, predmean, predvar, use_diag=True)
