from func2d import forrester2008
import numpy as np
import torch
import matplotlib.pyplot as plt
from test_general import test_single
plt.style.use(['science', 'grid'])
torch.set_default_dtype(torch.double)

from pathlib import Path

DIR = r'code/test_results/surmise_MVGP_MVIP/2dExample_separate/'
Path(DIR).mkdir(parents=True, exist_ok=True)

def test_n(n, noiseconst, save=False):
    x = np.random.uniform(0, 1, n)

    x = np.sort(x)
    f = forrester2008(x, noisy=True, noiseconst=noiseconst)

    xtest = np.linspace(0, 1, 100)
    ftest0 = forrester2008(xtest, noisy=False)
    ftest = forrester2008(xtest, noisy=True, noiseconst=noiseconst)

    x = torch.tensor(x).unsqueeze(1)
    f = torch.tensor(f)
    xtest = torch.tensor(xtest).unsqueeze(1)
    ftest = torch.tensor(ftest)
    ftest0 = torch.tensor(ftest0)

    emu, emumean, emustd, emupcto, \
        n, xtr, ftr, xte, fte = test_single(method='surmise', n=n, seed=0, noiseconst=noiseconst,
                                            ftr=f, xtr=x, fte=ftest, xte=xtest,
                                            fte0=ftest0, output_csv=True, dir=DIR,
                                            return_quant=True)

    model, predmean, predstd, Phi, \
        _, _, _, _, _ = test_single(method='MVGP', n=n, seed=0, kap=1, noiseconst=noiseconst,
                                            ftr=f, xtr=x, fte=ftest, xte=xtest,
                                            fte0=ftest0, output_csv=True, dir=DIR,
                                            return_quant=True)

    modelsp, predmeansp, predstdsp, Phisp, \
        _, _, _, _, _ = test_single(method='MVIP', n=n, ip_frac=0.5, seed=0, kap=1,
                                    noiseconst=noiseconst,
                                            ftr=f, xtr=x, fte=ftest, xte=xtest,
                                            fte0=ftest0, output_csv=True, dir=DIR,
                                            return_quant=True)

    plot(noiseconst, n, emupcto[:, 0] / (emu._info['pcw'][0] / np.sqrt(n)), emumean, emustd,
         model, predmean, predstd,
         modelsp, predmeansp, predstdsp,
         x, f,
         xtest, ftest0,
         save=save)

    return

def plot(noiseconst, n, emupct, emumean, emustd,
         model, predmean, predstd,
         modelsp, predmeansp, predstdsp,
         x, f,
         xtest, ftest,
         save=True):
    lw = 3
    ymax = np.max((np.max((emupct.T @ (emumean + 2*emustd))),
                   np.max(((model.Phi[:, 0] / model.pcw).T @ (predmean + 2*predstd)).numpy())))
    ymin = np.min((np.min((emupct.T @ (emumean - 2*emustd))),
                   np.min(((model.Phi[:, 0] / model.pcw).T @ (predmean - 2*predstd)).numpy())))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey='all')
    ax[0].plot(xtest, (emupct.T @ ftest.numpy()).T, linewidth=lw, label='True', color='k')
    ax[0].scatter(x, (emupct.T @ f.numpy()).T, label='Data', color='gray')
    ax[0].plot(xtest, (emupct.T @ emumean).T, linewidth=lw, label='Prediction', color='r')
    ax[0].plot(xtest, (emupct.T @ (emumean + 2*emustd)).T, linewidth=lw, linestyle='--', color='r')
    ax[0].plot(xtest, (emupct.T @ (emumean - 2*emustd)).T, linewidth=lw, linestyle='--', color='r')
    ax[0].set_title('PCGP', fontsize=30)

    ax[1].plot(xtest, ((model.Phi[:, 0] / model.pcw).T @ ftest).T, linewidth=lw, label='True', color='k')
    ax[1].scatter(x, (model.M * np.sqrt(n) / model.pcw).T, label='Data', color='gray')
    ax[1].plot(xtest, ((model.Phi[:, 0] / model.pcw).T @ predmean).T, linewidth=lw, label='Prediction', color='r', alpha=0.6)
    ax[1].plot(xtest, ((model.Phi[:, 0] / model.pcw).T @ (predmean + 2*predstd)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    ax[1].plot(xtest, ((model.Phi[:, 0] / model.pcw).T @ (predmean - 2*predstd)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    ax[1].set_title('PCGP w/ VI', fontsize=30)

    ax[2].plot(xtest, ((modelsp.Phi[:, 0] / modelsp.pcw).T @ ftest).T, linewidth=lw, label='True', color='k')
    ax[2].scatter(x, (modelsp.M * np.sqrt(n) / model.pcw).T, label='Data', color='gray')
    ax[2].plot(xtest, ((modelsp.Phi[:, 0] / modelsp.pcw).T @ predmeansp).T, linewidth=lw, label='Prediction', color='r', alpha=0.6)
    ax[2].plot(xtest, ((modelsp.Phi[:, 0] / modelsp.pcw).T @ (predmeansp + 2*predstdsp)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    ax[2].plot(xtest, ((modelsp.Phi[:, 0] / modelsp.pcw).T @ (predmeansp - 2*predstdsp)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    ax[2].set_title(r'PCGP w/ VI ($n = 2p$)', fontsize=30)

    for axi in ax.flatten():
        # axi.set(ylim=(ymin, ymax))
        axi.set_xlabel(r'$\boldsymbol x$', fontsize=20)
        axi.set_ylabel(r'$g(\boldsymbol x)$', fontsize=20, rotation=0,
                       labelpad=12)
        axi.tick_params(axis='both', which='major', labelsize=20)
    plt.suptitle('Noise stdev multiplier = {:d}'.format(noiseconst))
    plt.tight_layout()
    # plt.legend(fontsize=18)
    if save:
        plt.savefig('compare_{:d}_{:d}.png'.format(n, noiseconst), dpi=300)

    plt.close()


if __name__ == '__main__':
    # ns = [25, 50, 100, 250]  # 25,
    res = []
    for i in range(1):
        if i == 0:
            save = True
        else:
            save = False
        for noise in [5]:
            test_n(n=100, noiseconst=noise, save=True)