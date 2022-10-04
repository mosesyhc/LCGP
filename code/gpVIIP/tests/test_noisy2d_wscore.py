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
    # 0, 100
    # np.random.seed(50)
    x = np.random.uniform(0, 1, n)
    x = np.linspace(0, 1, n)

    x = np.sort(x)
    f = forrester2008(x, noisy=True, noiseconst=noiseconst)

    xtest = np.linspace(0, 1, 90)
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
        _, _, _, _, _ = test_single(method='MVIP', n=n, ip_frac=1, seed=0, kap=1,
                                    noiseconst=noiseconst,
                                            ftr=f, xtr=x, fte=ftest, xte=xtest,
                                            fte0=ftest0, output_csv=True, dir=DIR,
                                            return_quant=True)

    # if False:

    if True:
        fig, axes = plt.subplots(2, 1, figsize=(5, 8))
        axes[0].scatter(x=xtr, y=ftr[0], color='gray', s=8, label='Data')
        axes[0].scatter(x=xtr, y=ftr[1], color='gray', marker='D', s=8)
        axes[0].plot(xte, predmean.T, label='Full VI', color='b', linewidth=2)
        axes[0].plot(xte, emumean.T, label='GP', color='violet', linewidth=2)
        axes[0].plot(xte, predmeansp.T, label='VI w/ n={:d}p'.format(int(modelsp.n/modelsp.p)), color='r', linewidth=2)
        axes[0].set_xlabel(r'$x$')
        axes[0].set_ylabel('Mean')
        axes[0].set_title('Predictive Mean')

        axes[1].plot(xte, predstd.T, label='Full VI', color='b', linewidth=2)
        axes[1].plot(xte, emustd.T, label='GP', color='violet', linewidth=2)
        axes[1].plot(xte, predstdsp.T, label='VI w/ n={:d}p'.format(int(modelsp.n/modelsp.p)), color='r', linewidth=2)
        axes[1].set_xlabel(r'$x$')
        axes[1].set_ylabel('Stdev')
        axes[1].set_title('Predictive Standard Deviation')

        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels)

    # plot(noiseconst, n, emupcto[:, 0] / (emu._info['pcw'][0]), emumean, emustd,
    #      model, predmean, predstd,
    #      modelsp, predmeansp, predstdsp,
    #      x, f,
    #      xtest, ftest0,
    #      save=save)

    return model, modelsp

def plot(noiseconst, n, emupct, emumean, emustd,
         model, predmean, predstd,
         modelsp, predmeansp, predstdsp,
         x, f,
         xtest, ftest,
         save=True):
    lw = 3
    modelpct = model.Phi[:, 0] / (model.pcw * np.sqrt(model.n))
    modelsppct = modelsp.Phi[:, 0] / (modelsp.pcw * np.sqrt(modelsp.n))
    ymax = np.max((np.max((emupct.T @ (emumean + 2*emustd))),
                   np.max((modelpct.T @ (predmean + 2*predstd)).numpy())))
    ymin = np.min((np.min((emupct.T @ (emumean - 2*emustd))),
                   np.min((modelpct @ (predmean - 2*predstd)).numpy())))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey='all')
    ax[0].plot(xtest, (emupct.T @ ftest.numpy()).T, linewidth=lw, label='True', color='k')
    ax[0].scatter(x, (emupct.T @ f.numpy()).T, label='Data', color='gray')
    ax[0].plot(xtest, (emupct.T @ emumean).T, linewidth=lw, label='Prediction', color='r')
    ax[0].plot(xtest, (emupct.T @ (emumean + 2*emustd)).T, linewidth=lw, linestyle='--', color='r')
    ax[0].plot(xtest, (emupct.T @ (emumean - 2*emustd)).T, linewidth=lw, linestyle='--', color='r')
    ax[0].set_title('PCGP', fontsize=30)

    ax[1].plot(xtest, (modelpct.T @ ftest).T, linewidth=lw, label='True', color='k')
    ax[1].scatter(x, (model.M / model.pcw).T, label='Data', color='gray')
    ax[1].plot(xtest, (modelpct.T @ predmean).T, linewidth=lw, label='Prediction', color='r', alpha=0.6)
    ax[1].plot(xtest, (modelpct.T @ (predmean + 2*predstd)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    ax[1].plot(xtest, (modelpct.T @ (predmean - 2*predstd)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    ax[1].set_title('PCGP w/ VI', fontsize=30)

    ax[2].plot(xtest, (modelsppct.T @ ftest).T, linewidth=lw, label='True', color='k')
    ax[2].scatter(x, (modelsp.M / model.pcw).T, label='Data', color='gray')
    ax[2].plot(xtest, (modelsppct.T @ predmeansp).T, linewidth=lw, label='Prediction', color='r', alpha=0.6)
    ax[2].plot(xtest, (modelsppct.T @ (predmeansp + 2*predstdsp)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    ax[2].plot(xtest, (modelsppct.T @ (predmeansp - 2*predstdsp)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
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

    # plt.close()


if __name__ == '__main__':
    ns = [25] #, 50, 100] #, 250, 500] #, 250]  # 25,
    res = []
    res_sp = []
    for i in range(1):
        save = False
        for n in ns:
            m, ms = test_n(n=n, noiseconst=2, save=False)
            res.append(m)
            res_sp.append(ms)

    def print_diagnostics(m):
        _, _, lnug = m.parameter_clamp(m.lLmb, m.lsigma2, m.lnugGPs)
        testGPvar1, testGPvar2 = m.GPvarterm1.mean().item(), m.GPvarterm2.mean().item()
        _ = m.predictcov(m.theta)
        trainGPvar1, trainGPvar2 = m.GPvarterm1.mean().item(), m.GPvarterm2.mean().item()
        print('{:<5d} {:<5.3f} {:<5.3f} {:<6.6f} {:<6.6f} {:<6.6f} {:<6.6f}'.format(m.n, m.tau2gps.item(), lnug.item(), trainGPvar1, trainGPvar2, testGPvar1, testGPvar2))

    header = ['n', 'tau2gps', 'lnug', 'Gvar', 'train_term1', 'train_term2', 'test_term1', 'test_term2']
    print('Full VI')
    print('{:<5s} {:<5s} {:<5s} {:<5s} {:<6s} {:<6s} {:<6s} {:<6s}'.format(*header))
    for m in res:
        print_diagnostics(m)

    print('IP')
    print('{:<5s} {:<5s} {:<5s} {:<5s}  {:<6s} {:<6s} {:<6s} {:<6s}'.format(*header))
    for m in res_sp:
        print_diagnostics(m)