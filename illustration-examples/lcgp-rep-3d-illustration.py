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

def make_hartmann_tf(alpha, A, P, a=0.0, b=1.0):
    A_tf = tf.constant(A, dtype=tf.float32)       
    P_tf = tf.constant(P, dtype=tf.float32)      
    alpha_tf = tf.constant(alpha, dtype=tf.float32)
    a_tf = tf.constant(a, dtype=tf.float32)
    b_tf = tf.constant(b, dtype=tf.float32)

    def hartmann(x):
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32) 
        x_exp = tf.expand_dims(x_tf, axis=1)              
        r = tf.reduce_sum(A_tf * tf.square(x_exp - P_tf), axis=-1)  
        val = (a_tf - tf.reduce_sum(alpha_tf * tf.exp(-r), axis=-1)) / b_tf
        return val
    return hartmann

alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float32)
A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
              [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
              [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
              [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]], dtype=np.float32)
P = (1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                      [2329, 4135, 8307, 3736, 1004, 9991],
                      [2348, 1451, 3522, 2883, 3047, 6650],
                      [4047, 8828, 8732, 5743, 1091,  381]], dtype=np.float32))

hartmann6d = make_hartmann_tf(alpha, A, P, a=0.0, b=1.0)

def generate_hartmann_data(x, noisy=True, noise_level=0.01):
    """
    x: (n, 6)
    returns y with shape (3, n)
    """
    x = np.asarray(x, dtype=np.float32)
    y_base = hartmann6d(x).numpy()

    y1 = y_base
    y2 = 0.5 * y_base + 2.0 * np.sum(x, axis=1) - 1.0
    y3 = -0.8 * y_base - 3.0 * np.prod(x, axis=1) + 2.0

    if noisy:
        y1 = y1 + np.random.normal(0, noise_level * 0.1, y1.shape)
        y2 = y2 + np.random.normal(0, noise_level * 0.2, y2.shape)
        y3 = y3 + np.random.normal(0, noise_level * 0.3, y3.shape)

    y = np.vstack((y1, y2, y3)) 
    return y

ns = [50, 100]   
ntest = 500
results_fig_path = './results_figure_rep/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)

results_rmse = {}

for n in ns:
    print(f'Testing number of samples: {n}')
    xtrain = np.random.uniform(0, 1, (n, 6)).astype(np.float32)
    xtest = np.random.uniform(0, 1, (ntest, 6)).astype(np.float32)

    ytrue = generate_hartmann_data(xtest, noisy=False)  
    generating_noises_var = np.array([0.005, 0.1, 0.3]) * ytrue.var(1)

    ytrain = generate_hartmann_data(xtrain, noisy=True, noise_level=0.01) 
    ytest = generate_hartmann_data(xtest, noisy=True, noise_level=0.01)   

    data = {
        'xtrain': xtrain,  
        'xtest': xtest, 
        'ytrain': ytrain,  
        'ytest': ytest,   
        'ytrue': ytrue   
    }

    modelrun = LCGPRun(
        runno=f'n{n}_rep',
        data=data,
        num_latent=ytrain.shape[0]-1,
        submethod='rep'   
    )
    modelrun.define_model()

    t0 = time.time()
    modelrun.train()
    t1 = time.time()

    predmean, predvar = modelrun.predict()  

    rmse = evaluation.rmse(ytrue, predmean)
    nrmse = evaluation.normalized_rmse(ytrue, predmean)
    pcover, pwidth = evaluation.intervalstats(ytest, predmean, predvar)
    dss = evaluation.dss(ytrue, predmean, predvar, use_diag=True)

    result = {
        'modelname': modelrun.modelname,
        'modelrun': modelrun.runno,
        'n': modelrun.n,
        'traintime': t1 - t0,
        'rmse': rmse,
        'nrmse': nrmse,
        'pcover': pcover,
        'pwidth': pwidth,
        'dss': dss
    }

    df = pd.DataFrame.from_dict(result, orient='index').reset_index()
    print(df)

    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    order_test = np.argsort(xtest[:, 0])
    order_train = np.argsort(xtrain[:, 0])

    for i in range(3):
        ax[i].scatter(xtrain[order_train, 0], ytrain[i, order_train],
                      label='Training Data', s=10, alpha=0.6)
        ax[i].plot(xtest[order_test, 0], ytrue[i, order_test],
                   label='True', linewidth=1.5)
        ax[i].plot(xtest[order_test, 0], predmean[i, order_test],
                   label='Prediction', linewidth=1.2)
        ax[i].fill_between(
            xtest[order_test, 0],
            predmean[i, order_test] - 2.0 * np.sqrt(predvar[i, order_test]),
            predmean[i, order_test] + 2.0 * np.sqrt(predvar[i, order_test]),
            alpha=0.25, label='95% CI'
        )
        ax[i].set_ylabel(f'Output {i+1}')
        if i == 0:
            ax[i].legend(loc='best', fontsize=8)

    ax[-1].set_xlabel('x[:, 0]')
    plt.tight_layout()
    figpath = Path(results_fig_path) / f'lcgp-rep-n-{n}.png'
    plt.savefig(figpath, dpi=150)
    plt.close(fig)

    results_rmse[n] = rmse

print("RMSE summary:", results_rmse)
