import numpy as np
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from lcgp import evaluation
from call_model import LCGPRun

np.random.seed(42)
tf.random.set_seed(42)

def f_true(x):  
    x = np.asarray(x, dtype=np.float64)
    f1 = 0.8 + 0.3*np.sin(2*np.pi*x) + 0.2*x
    f2 = 0.3 + 0.5*np.cos(2*np.pi*x)
    f3 = -0.4 - (x-0.5)**2 + 0.2*np.sin(4*np.pi*x)
    return np.vstack([f1, f2, f3]) 

def make_rep_data(n_unique=12, rep_choices=(1,2,3,4), noise_std=(0.05, 0.08, 0.10)):
    """
    Returns:
      xtrain: (N, 1) stacked with replicates
      ytrain: (3, N) noisy
      xtest:  (T, 1)
      ytrue:  (3, T)
    """
    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)
    r = np.random.choice(rep_choices, size=n_unique, replace=True)

    xs = []
    ys = []
    for i, xi in enumerate(x_unique):
        yi_true = f_true([xi])[:, 0]  
        for _ in range(r[i]):
            eps = np.array([
                np.random.normal(0, noise_std[0]),
                np.random.normal(0, noise_std[1]),
                np.random.normal(0, noise_std[2]),
            ], dtype=np.float64)
            xs.append([xi])
            ys.append(yi_true + eps)

    xtrain = np.array(xs, dtype=np.float64)              
    ytrain = np.array(ys, dtype=np.float64).T         

    xtest = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]  
    ytrue = f_true(xtest[:, 0])                                    
    return xtrain, ytrain, xtest, ytrue

results_fig_path = './results_figure_rep_1d/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)

xtrain, ytrain, xtest, ytrue = make_rep_data(
    n_unique=12,
    rep_choices=(1,2,3,4,5),
    noise_std=(0.05, 0.08, 0.10)
)

data = {
    'xtrain': xtrain,         
    'xtest': xtest,          
    'ytrain': ytrain,         
    'ytest':  ytrue,          
    'ytrue':  ytrue
}

num_latent = ytrain.shape[0] - 1   
modelrun = LCGPRun(
    runno='rep_1d',
    data=data,
    num_latent=num_latent,
    submethod='rep'   
)
modelrun.define_model()

t0 = time.time()
modelrun.train()
t1 = time.time()

predmean, predvar = modelrun.predict()  

rmse  = evaluation.rmse(ytrue, predmean)
nrmse = evaluation.normalized_rmse(ytrue, predmean)
pcover, pwidth = evaluation.intervalstats(ytrue, predmean, predvar)  
dss = evaluation.dss(ytrue, predmean, predvar, use_diag=True)

print("train time (s):", t1 - t0)
print("RMSE:", rmse)
print("NRMSE:", nrmse)
print("PI coverage (≈95% target):", pcover)
print("PI width:", pwidth)
print("DSS:", dss)

fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

order_test  = np.argsort(xtest[:, 0])
order_train = np.argsort(xtrain[:, 0])

for i in range(3):
    # replicates as points
    ax[i].scatter(xtrain[order_train, 0], ytrain[i, order_train],
                  s=12, alpha=0.65, label='replicates' if i == 0 else None)
    # true curve
    ax[i].plot(xtest[order_test, 0], ytrue[i, order_test],
               lw=1.8, label='true' if i == 0 else None)
    # prediction mean
    # ax[i].plot(xtest[order_test, 0], predmean[i, order_test],
    #            lw=1.5, label='LCGP mean' if i == 0 else None)
    # # 95% predictive interval
    # lo = predmean[i, order_test] - 1.96*np.sqrt(predvar[i, order_test])
    # hi = predmean[i, order_test] + 1.96*np.sqrt(predvar[i, order_test])
    # ax[i].fill_between(xtest[order_test, 0], lo, hi, alpha=0.22,
    #                    label='95% pred. interval' if i == 0 else None)

    ax[i].set_ylabel(f'$f_{i+1}(x)$')

ax[-1].set_xlabel('x')
ax[0].legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(Path(results_fig_path)/'lcgp_rep_1d_demo.png', dpi=150)
plt.close()
