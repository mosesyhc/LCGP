import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from lcgp.lcgp_rep import LCGP
from debug.data_generation.read_debug_data import load_train_test_csv

np.random.seed(42)
tf.random.set_seed(42)

RESULTS_DIR = Path("debug/results_rep_plots/")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = "debug/data_generation/data/train_data.csv"
TEST_CSV  = "debug/data_generation/data/test_data.csv"

NUM_OUTPUTS = 3
PLOT_MODE = "y" 

BAND_KIND = "conf" 

REP_STANDARDIZE_YBAR = False

data = load_train_test_csv(TRAIN_CSV, TEST_CSV)
xtrain = np.asarray(data["xtrain"], dtype=np.float64)      
ytrain = np.asarray(data["ytrain"], dtype=np.float64)      
xtest  = np.asarray(data["xtest"],  dtype=np.float64)       

ytrue = data.get("ytrue", None)
if ytrue is not None:
    ytrue = np.asarray(ytrue, dtype=np.float64)          

order_test  = np.argsort(xtest[:, 0])
order_train = np.argsort(xtrain[:, 0])

mdl = LCGP(
    y=ytrain,
    x=xtrain,
    q=None,
    var_threshold=None,
    submethod="rep",
    diag_error_structure=[1, 1, 1],
    robust_mean=True,
    rep_standardize_ybar=REP_STANDARDIZE_YBAR,
    verbose=True,
)

mdl.fit(verbose=True)

predmean_tf, ypredvar_tf, yconfvar_tf = mdl.predict(x0=xtest, return_fullcov=False)
predmean = predmean_tf.numpy()    
ypredvar = ypredvar_tf.numpy()   
yconfvar = yconfvar_tf.numpy()   

band_var = yconfvar if BAND_KIND == "conf" else ypredvar

if PLOT_MODE == "g":
    ghat_test = np.asarray(mdl.ghat)           
    gstd_test = np.sqrt(np.asarray(mdl.gvar))  
    q = ghat_test.shape[0]

    x_unique = np.asarray(getattr(mdl, "x_unique", None))
    ghat_tr  = np.asarray(getattr(mdl, "mks", None))

    fig, axes = plt.subplots(q, 1, figsize=(10, 1.9 * q), sharex=True)
    if q == 1:
        axes = [axes]

    x_test = xtest[order_test, 0]

    for k, ax in enumerate(axes):
        m = ghat_test[k, order_test]
        s = gstd_test[k, order_test]

        ax.plot(x_test, m, lw=1.8, label=fr"$g_{{{k+1}}}(x)$ mean")
        ax.fill_between(x_test, m - 1.96 * s, m + 1.96 * s, alpha=0.22, label="95% band")

        if x_unique is not None and ghat_tr is not None:
            xu = x_unique[:, 0]
            ord_u = np.argsort(xu)
            ax.scatter(xu[ord_u], ghat_tr[k, ord_u], s=12, alpha=0.65, label="train pts")

        ax.set_ylabel(fr"$g_{{{k+1}}}(x)$")
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lcgp_latents_gkx_with_points.png", dpi=150)
    plt.close()

else:


    fig, ax = plt.subplots(NUM_OUTPUTS, 1, figsize=(10, 7), sharex=True)

    x_tr = xtrain[order_train, 0]
    x_te = xtest[order_test, 0]

    for i in range(NUM_OUTPUTS):
        # replicates
        ax[i].scatter(
            x_tr,
            ytrain[i, order_train],
            s=12,
            alpha=0.65,
            label="replicates" if i == 0 else None,
        )

        # true
        if ytrue is not None:
            ax[i].plot(
                x_te,
                ytrue[i, order_test],
                lw=1.8,
                label="true" if i == 0 else None,
            )

        # mean
        ax[i].plot(
            x_te,
            predmean[i, order_test],
            lw=1.5,
            label="LCGP mean" if i == 0 else None,
        )

        # band
        s = np.sqrt(np.maximum(band_var[i, order_test], 0.0))
        lo = predmean[i, order_test] - 1.96 * s
        hi = predmean[i, order_test] + 1.96 * s

        ax[i].fill_between(
            x_te,
            lo,
            hi,
            alpha=0.22,
            label=("95% credible band" if BAND_KIND == "conf" else "95% predictive band") if i == 0 else None,
        )

        ax[i].set_ylabel(f"$f_{i+1}(x)$")

    ax[-1].set_xlabel("x")
    ax[0].legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_name = "lcgp_rep_1d_demo.png" if BAND_KIND == "conf" else "lcgp_rep_1d_demo_predband.png"
    plt.savefig(RESULTS_DIR / out_name, dpi=150)
    plt.close()