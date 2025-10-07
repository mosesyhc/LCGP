import numpy as np
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from call_model import LCGPRun
from lcgp import evaluation

np.random.seed(42)
tf.random.set_seed(42)

def f_true(x):
    x = np.asarray(x, dtype=np.float64)
    f1 = 0.8 + 0.3*np.sin(2*np.pi*x) + 0.2*x
    f2 = 0.3 + 0.5*np.cos(2*np.pi*x)
    f3 = -0.4 - (x-0.5)**2 + 0.2*np.sin(4*np.pi*x)
    return np.vstack([f1, f2, f3])

def make_rep_data(n_unique=12, rep_choices=(1,2,3,4), noise_std=(0.05, 0.08, 0.10), seed=None):
    """
    Returns:
      xtrain: (N, 1) stacked with replicates
      ytrain: (3, N) noisy
      xtest:  (T, 1)
      ytrue:  (3, T)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        choice = lambda a, size=None, replace=True: rng.choice(a, size=size, replace=replace)
        normal = lambda mu, sig: rng.normal(mu, sig)
    else:
        choice = lambda a, size=None, replace=True: np.random.choice(a, size=size, replace=replace)
        normal = lambda mu, sig: np.random.normal(mu, sig)

    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)
    r = choice(rep_choices, size=n_unique, replace=True)

    xs, ys = [], []
    for i, xi in enumerate(x_unique):
        yi_true = f_true([xi])[:, 0] 
        for _ in range(int(r[i])):
            eps = np.array([
                normal(0, noise_std[0]),
                normal(0, noise_std[1]),
                normal(0, noise_std[2]),
            ], dtype=np.float64)
            xs.append([xi])
            ys.append(yi_true + eps)

    xtrain = np.array(xs, dtype=np.float64)           
    ytrain = np.array(ys, dtype=np.float64).T          
    xtest  = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]
    ytrue  = f_true(xtest[:, 0])                  
    return xtrain, ytrain, xtest, ytrue

def make_rep_data_skewed(n_unique=40,
                         heavy_region=(0.20, 0.45),
                         light_rep_choices=(1, 2),
                         heavy_rep_choices=(8, 12, 16, 20),
                         noise_std=(0.05, 0.08, 0.10),
                         seed=None):
    """
    many replicates inside heavy_region, few elsewhere.
    """
    rng = np.random.default_rng(seed)
    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)

    xs, ys = [], []
    for xi in x_unique:
        if heavy_region[0] <= xi <= heavy_region[1]:
            r = int(rng.choice(heavy_rep_choices))
        else:
            r = int(rng.choice(light_rep_choices))

        yi_base = f_true([xi])[:, 0]  # (3,)
        for _ in range(r):
            eps = np.array([
                rng.normal(0, noise_std[0]),
                rng.normal(0, noise_std[1]),
                rng.normal(0, noise_std[2]),
            ], dtype=np.float64)
            xs.append([xi])
            ys.append(yi_base + eps)

    xtrain = np.array(xs, dtype=np.float64)           
    ytrain = np.array(ys, dtype=np.float64).T          
    xtest  = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]
    ytrue  = f_true(xtest[:, 0])                     
    return xtrain, ytrain, xtest, ytrue

def make_rep_data_hotspots(n_unique=50,
                           hotspots=((0.15, 10, 15), (0.50, 18, 25), (0.80, 12, 20)),
                           base_rep_choices=(1,),
                           noise_std=(0.05, 0.08, 0.10),
                           seed=None):
    """
    a few 'hot-spot' x locations with very high replication.
    """
    rng = np.random.default_rng(seed)
    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)

    hotspot_idx = {}
    for (x0, lo, hi) in hotspots:
        idx = np.argmin(np.abs(x_unique - x0))
        hotspot_idx[idx] = (lo, hi)

    xs, ys = [], []
    for i, xi in enumerate(x_unique):
        if i in hotspot_idx:
            lo, hi = hotspot_idx[i]
            r = int(rng.integers(lo, hi + 1))
        else:
            r = int(rng.choice(base_rep_choices))

        yi_base = f_true([xi])[:, 0]
        for _ in range(r):
            eps = np.array([
                rng.normal(0, noise_std[0]),
                rng.normal(0, noise_std[1]),
                rng.normal(0, noise_std[2]),
            ], dtype=np.float64)
            xs.append([xi])
            ys.append(yi_base + eps)

    xtrain = np.array(xs, dtype=np.float64)
    ytrain = np.array(ys, dtype=np.float64).T
    xtest  = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]
    ytrue  = f_true(xtest[:, 0])
    return xtrain, ytrain, xtest, ytrue

# --- CASE 1: Uniform-ish replication ---
results_fig_path = './results_figure_rep_1d_uniform/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)
xtrain, ytrain, xtest, ytrue = make_rep_data(
    n_unique=16,
    rep_choices=(1,2,3,4,5),
    noise_std=(0.05, 0.08, 0.10),
    seed=2025
)

# --- CASE 2: Skewed replication ---
# results_fig_path = './results_figure_rep_1d_skewed/'
# Path(results_fig_path).mkdir(parents=True, exist_ok=True)
# xtrain, ytrain, xtest, ytrue = make_rep_data_skewed(
#     n_unique=40,
#     heavy_region=(0.20, 0.45),
#     light_rep_choices=(1, 2),
#     heavy_rep_choices=(8, 12, 16, 20),
#     noise_std=(0.05, 0.08, 0.10),
#     seed=123
# )

# # --- CASE 3: Hot-spots ---
# results_fig_path = './results_figure_rep_1d_hotspots/'
# Path(results_fig_path).mkdir(parents=True, exist_ok=True)
# xtrain, ytrain, xtest, ytrue = make_rep_data_hotspots(
#     n_unique=50,
#     hotspots=((0.15, 10, 15), (0.50, 18, 25), (0.80, 12, 20)),
#     base_rep_choices=(1,),
#     noise_std=(0.05, 0.08, 0.10),
#     seed=7
# )

data = {
    'xtrain': xtrain,
    'xtest':  xtest,
    'ytrain': ytrain,
    'ytest':  ytrue,  
    'ytrue':  ytrue
}

modelrun = LCGPRun(
    runno='rep_1d',
    data=data,
    num_latent=3,        
    var_threshold=None,  
    submethod='full',
    diag_error_structure=[1,1,1],
    robust_mean=True,
)
modelrun.define_model()

t0 = time.time()
modelrun.train()
t1 = time.time()

predmean, ypredvar, yconfvar = modelrun.predict(return_fullcov=False)

rmse  = evaluation.rmse(ytrue, predmean)
nrmse = evaluation.normalized_rmse(ytrue, predmean)
pcover, pwidth = evaluation.intervalstats(ytrue, predmean, yconfvar) 
dss = evaluation.dss(ytrue, predmean, yconfvar, use_diag=True)

print("train time (s):", round(t1 - t0, 3))
print("RMSE:", rmse)
print("NRMSE:", nrmse)
print("95% PI coverage:", pcover)
print("95% PI width:", pwidth)
print("DSS:", dss)

fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
order_test  = np.argsort(xtest[:, 0])
order_train = np.argsort(xtrain[:, 0])

for i in range(3):
    ax[i].scatter(xtrain[order_train, 0], ytrain[i, order_train],
                  s=12, alpha=0.65, label='replicates' if i == 0 else None)

    ax[i].plot(xtest[order_test, 0], ytrue[i, order_test],
               lw=1.8, label='true' if i == 0 else None)

    ax[i].plot(xtest[order_test, 0], predmean[i, order_test],
               lw=1.5, label='LCGP mean' if i == 0 else None)

    lo = predmean[i, order_test] - 1.96*np.sqrt(yconfvar[i, order_test])
    hi = predmean[i, order_test] + 1.96*np.sqrt(yconfvar[i, order_test])
    ax[i].fill_between(xtest[order_test, 0], lo, hi, alpha=0.22,
                       label='95% credible band' if i == 0 else None)

    ax[i].set_ylabel(f'$f_{i+1}(x)$')

ax[-1].set_xlabel('x')
ax[0].legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(Path(results_fig_path)/'lcgp_rep_1d_demo.png', dpi=150)
plt.close()
