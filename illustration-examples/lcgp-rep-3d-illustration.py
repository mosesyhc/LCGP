import numpy as np
import tensorflow as tf
import time
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
            eps = np.array([normal(0, noise_std[0]),
                            normal(0, noise_std[1]),
                            normal(0, noise_std[2])], dtype=np.float64)
            xs.append([xi]); ys.append(yi_true + eps)

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
    rng = np.random.default_rng(seed)
    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)

    xs, ys = [], []
    for xi in x_unique:
        r = int(rng.choice(heavy_rep_choices) if (heavy_region[0] <= xi <= heavy_region[1])
                else rng.choice(light_rep_choices))
        yi_base = f_true([xi])[:, 0]
        for _ in range(r):
            eps = np.array([rng.normal(0, noise_std[0]),
                            rng.normal(0, noise_std[1]),
                            rng.normal(0, noise_std[2])], dtype=np.float64)
            xs.append([xi]); ys.append(yi_base + eps)

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
    rng = np.random.default_rng(seed)
    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)
    hotspot_idx = {np.argmin(np.abs(x_unique - x0)): (lo, hi) for (x0, lo, hi) in hotspots}

    xs, ys = [], []
    for i, xi in enumerate(x_unique):
        if i in hotspot_idx:
            lo, hi = hotspot_idx[i]
            r = int(rng.integers(lo, hi + 1))
        else:
            r = int(rng.choice(base_rep_choices))
        yi_base = f_true([xi])[:, 0]
        for _ in range(r):
            eps = np.array([rng.normal(0, noise_std[0]),
                            rng.normal(0, noise_std[1]),
                            rng.normal(0, noise_std[2])], dtype=np.float64)
            xs.append([xi]); ys.append(yi_base + eps)

    xtrain = np.array(xs, dtype=np.float64)
    ytrain = np.array(ys, dtype=np.float64).T
    xtest  = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]
    ytrue  = f_true(xtest[:, 0])
    return xtrain, ytrain, xtest, ytrue

# --------------------------
# Choose dataset
# --------------------------
# CASE 1: Uniform-ish replication
# results_fig_path = './results_figure_rep_1d_uniform/'
# Path(results_fig_path).mkdir(parents=True, exist_ok=True)
# xtrain, ytrain, xtest, ytrue = make_rep_data(
#     n_unique=16,
#     rep_choices=(1,2,3,4,5),
#     noise_std=(0.05, 0.08, 0.10),
#     seed=2025
# )

# # CASE 2: Skewed replication
results_fig_path = './results_figure_rep_1d_skewed/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)
xtrain, ytrain, xtest, ytrue = make_rep_data_skewed(
    n_unique=40,
    heavy_region=(0.20, 0.45),
    light_rep_choices=(1, 2),
    heavy_rep_choices=(8, 12, 16, 20),
    noise_std=(0.05, 0.08, 0.10),
    seed=123
)

# CASE 3: Hot-spots
# results_fig_path = './results_figure_rep_1d_hotspots/'
# Path(results_fig_path).mkdir(parents=True, exist_ok=True)
# xtrain, ytrain, xtest, ytrue = make_rep_data_hotspots(
#     n_unique=50,
#     hotspots=((0.15, 10, 15), (0.50, 18, 25), (0.80, 12, 20)),
#     base_rep_choices=(1,),
#     noise_std=(0.05, 0.08, 0.10),
#     seed=7
# )

# --------------------------
# Build & train the model
# --------------------------
SUBMETHOD = 'rep'   # set 'rep' for replicated data; 'full' for non-replicated
PLOT_MODE = 'y'     # 'g' to plot latents; 'y' to plot outputs

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
    submethod=SUBMETHOD,
    diag_error_structure=[1,1,1],
    robust_mean=True,
)
modelrun.define_model()

t0 = time.time()
modelrun.train()
t1 = time.time()

predmean, ypredvar, yconfvar = modelrun.predict(return_fullcov=False)

# --------------------------
# Transform consistency check
# --------------------------
def transform_consistency_check(modelrun, predmean_from_runner, xtest):
    mdl = getattr(modelrun, 'model', None) or getattr(modelrun, 'lcgp', None)
    assert mdl is not None, "Couldn't find underlying LCGP model."

    _ = mdl.predict(x0=xtest, return_fullcov=False)

    lLmb, lLmb0, built_lsigma2s, lnug = mdl.get_param()
    sigma_sqrt = tf.sqrt(tf.exp(built_lsigma2s)).numpy()
    phi = mdl.phi.numpy()                               
    ghat = np.asarray(mdl.ghat)                         

    if mdl.submethod == 'rep':
        y_std = phi @ ghat    
        y_from_g = y_std * np.asarray(mdl.ybar_std) + np.asarray(mdl.ybar_mean)
    else:
        psi = phi * sigma_sqrt[:, None]                    
        predmean_std = psi @ ghat                          
        y_from_g = mdl.tx_y(predmean_std).numpy()          

    diff = np.max(np.abs(y_from_g - predmean_from_runner))
    print(f"[transform check] max |recomposed - runner| = {diff:.3e}")

transform_consistency_check(modelrun, predmean, xtest)

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

order_test  = np.argsort(xtest[:, 0])
order_train = np.argsort(xtrain[:, 0])

if PLOT_MODE == 'g':
    mdl = getattr(modelrun, 'model', None) or getattr(modelrun, 'lcgp', None)

    _ = mdl.predict(x0=xtest, return_fullcov=False)
    ghat_test = np.asarray(mdl.ghat)           
    gstd_test = np.sqrt(np.asarray(mdl.gvar))  

    # _ = mdl.predict(x0=xtrain, return_fullcov=False)
    xtrain = np.asarray(mdl.x_unique)
    order_train = np.argsort(xtrain[:, 0])
    ghat_tr = np.asarray(tf.transpose(tf.matmul((tf.transpose(mdl.ybar_s * tf.cast(mdl.r, tf.float64))
                                                 * tf.exp(-0.5 * mdl.lsigma2s)), mdl.phi)))

    # ghat_tr = np.asarray(mdl.ghat)               #  ghat_tr = Ybar Psi, Psi = Sigma^{1/2} Phi, [U^T Y = R Ybar]
    # gstd_tr = np.sqrt(np.asarray(mdl.gvar))

    q = ghat_test.shape[0]
    fig, axes = plt.subplots(q, 1, figsize=(10, 1.9*q), sharex=True)
    if q == 1: axes = [axes]

    x_test = xtest[order_test, 0]
    x_tr   = xtrain[order_train, 0]

    for k, ax in enumerate(axes):
        m = ghat_test[k, order_test]
        s = gstd_test[k, order_test]
        ax.plot(x_test, m, lw=1.8, label=fr'$g_{{{k+1}}}(x)$ mean')
        ax.fill_between(x_test, m - 1.96*s, m + 1.96*s, alpha=0.22, label='95% band')

        ax.scatter(x_tr, ghat_tr[k, order_train],
                   s=12, alpha=0.65, label='train pts')

        ax.set_ylabel(fr'$g_{{{k+1}}}(x)$')
        ax.legend(loc='best', fontsize=9)

    axes[-1].set_xlabel('x')
    plt.tight_layout()
    plt.savefig(Path(results_fig_path)/'lcgp_latents_gkx_with_points.png', dpi=150)
    plt.close()


else:
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
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
