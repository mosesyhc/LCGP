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
    # np.random.seed(50)
    x = np.random.uniform(0, 1, n)
    # x = np.linspace(0, 1, n)

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
        n, xtr, ftr, xte, fte = test_single(method='surmise', fname='forrester2d', n=n, seed=0, noiseconst=noiseconst,
                                            ftr=f, xtr=x, fte=ftest, xte=xtest,
                                            fte0=ftest0, output_csv=True, dir=DIR,
                                            return_quant=True)

    model, predmean, predstd, Phi, \
        _, _, _, _, _ = test_single(method='MVGP', fname='forrester2d', n=n, seed=0, # kap=1,
                                    noiseconst=noiseconst,
                                    ftr=f, xtr=x, fte=ftest, xte=xtest,
                                    fte0=ftest0, output_csv=True, dir=DIR,
                                    return_quant=True)
    #
    # model, predmean, predstd, Phi, \
    #     _, _, _, _, _ = test_single(method='MVIP', n=n, ip_frac=1/4, seed=0, # kap=1,
    #                                 noiseconst=noiseconst,
    #                                 ftr=f, xtr=x, fte=ftest, xte=xtest,
    #                                 fte0=ftest0, output_csv=True, dir=DIR,
    #                                 return_quant=True)


    if False:
        predub = (predmean + 2 * predstd)
        predlb = (predmean - 2 * predstd)
        emuub = (emumean + 2 * emustd)
        emulb = (emumean - 2 * emustd)

        fig, axes = plt.subplots(3, 1, figsize=(5, 10))
        axes[0].scatter(x=xtr, y=ftr[0], color='gray', s=8, label='Data')
        axes[0].plot(xte, predmean[0].T, label='Full VI', color='b', linewidth=2)
        axes[0].plot(xte, emumean[0].T, label='GP', color='violet', linewidth=2)
        axes[0].plot(xte, predub[0].T, label='Full VI', color='b', linestyle='--', linewidth=2)
        axes[0].plot(xte, emuub[0].T, label='GP', color='violet', linestyle='--', linewidth=2)
        axes[0].plot(xte, predlb[0].T, label='Full VI', color='b', linestyle='--', linewidth=2)
        axes[0].plot(xte, emulb[0].T, label='GP', color='violet', linestyle='--', linewidth=2)
        # axes[0].plot(xte, predmeansp.T, label='VI w/ n={:d}p'.format(int(modelsp.n/modelsp.p)), color='r', linewidth=2)
        axes[0].set_xlabel(r'$x$')
        axes[0].set_ylabel(r'$f_1(x)$')
        axes[0].set_title('Predictive Mean')

        axes[1].scatter(x=xtr, y=ftr[1], color='gray', s=8, label='Data')
        axes[1].plot(xte, predmean[1].T, label='Full VI', color='b', linewidth=2)
        axes[1].plot(xte, emumean[1].T, label='GP', color='violet', linewidth=2)
        axes[1].plot(xte, predub[1].T, label='Full VI', color='b', linestyle='--', linewidth=2)
        axes[1].plot(xte, emuub[1].T, label='GP', color='violet', linestyle='--', linewidth=2)
        axes[1].plot(xte, predlb[1].T, label='Full VI', color='b', linestyle='--', linewidth=2)
        axes[1].plot(xte, emulb[1].T, label='GP', color='violet', linestyle='--', linewidth=2)
        axes[1].set_xlabel(r'$x$')
        axes[1].set_ylabel(r'$f_2(x)$')
        axes[1].set_title('')


        axes[2].plot(xte, predstd.T, label='Full VI', color='b', linewidth=2)
        axes[2].plot(xte, emustd.T, label='GP', color='violet', linewidth=2)
        # axes[2].plot(xte, predstdsp.T, label='VI w/ n={:d}p'.format(int(modelsp.n/modelsp.p)), color='r', linewidth=2)
        axes[2].set_xlabel(r'$x$')
        axes[2].set_ylabel('Stdev')
        axes[2].set_title('Predictive Standard Deviation')

        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels)
        plt.tight_layout()
        plt.savefig('2dplots_{:d}.png'.format(xtr.shape[0]), dpi=75)

    # plot(noiseconst, n, emupcto[:, 0] / (emu._info['pcw'][0]), emumean, emustd,
    #      model, predmean, predstd,
    #      # modelsp, predmeansp, predstdsp,
    #      x, f,
    #      xtest, ftest0,
    #      save=save)

    return model  #, modelsp

def plot(noiseconst, n, emupct, emumean, emustd,
         model, predmean, predstd,
         # modelsp, predmeansp, predstdsp,
         x, f,
         xtest, ftest,
         save=True):
    lw = 3
    modelpct = model.pct[:, 0] / (model.pcw * np.sqrt(model.n))
    # modelsppct = modelsp.Phi[:, 0] / (modelsp.pcw * np.sqrt(modelsp.n))
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
    #
    # ax[2].plot(xtest, (modelsppct.T @ ftest).T, linewidth=lw, label='True', color='k')
    # ax[2].scatter(x, (modelsp.M / model.pcw).T, label='Data', color='gray')
    # ax[2].plot(xtest, (modelsppct.T @ predmeansp).T, linewidth=lw, label='Prediction', color='r', alpha=0.6)
    # ax[2].plot(xtest, (modelsppct.T @ (predmeansp + 2*predstdsp)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    # ax[2].plot(xtest, (modelsppct.T @ (predmeansp - 2*predstdsp)).T, linewidth=lw, linestyle='--', color='r', alpha=0.6)
    # ax[2].set_title(r'PCGP w/ VI ($n = 2p$)', fontsize=30)

    for axi in ax.flatten():
        # axi.set(ylim=(ymin, ymax))
        axi.set_xlabel(r'$\boldsymbol x$', fontsize=20)
        axi.set_ylabel(r'$g(\boldsymbol x)$', fontsize=20, rotation=0,
                       labelpad=12)
        axi.tick_params(axis='both', which='major', labelsize=20)
    plt.suptitle('Noise stdev multiplier = {:d}'.format(noiseconst))
    plt.tight_layout()
    plt.legend(fontsize=18)
    if save:
        plt.savefig('compare_{:d}_{:d}.png'.format(n, noiseconst), dpi=300)
    plt.show()
    # plt.close()


if __name__ == '__main__':
    ns = [50] #, 100, 250, 500] #, 1000]  # 25,
    res = []
    res_sp = []
    for i in range(1):
        save = False
        for n in ns:
            m = test_n(n=n, noiseconst=1, save=False)  # , ms
            res.append(m)
            # res_sp.append(ms)

    def print_diagnostics(m):
        lLmb, lsigma2, lnug, ltau = m.parameter_clamp(m.lLmb, m.ltau2GPs, m.lsigma2, m.lnugGPs)
        testGPvar1, testGPvar2 = m.GPvarterm1.mean().item(), m.GPvarterm2.mean().item()
        _ = m.predictcov(m.x)
        trainGPvar1, trainGPvar2 = m.GPvarterm1.mean().item(), m.GPvarterm2.mean().item()
        Gvar = m.G.var()
        Mvar = m.M.var()
        print('{:<6d} {:<6.3f} {:<6.3f} {:<6.3f} {:<6.3f} {:<6.3f} {:<6.3f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
            m.n, ltau[0].item(), lnug[0].item(), lLmb[0, -1].item(), lsigma2.item(), Gvar, Mvar, trainGPvar1, trainGPvar2, testGPvar1, testGPvar2))

    header = ['n', 'ltau2gps', 'lnug', 'llmb[-1]', 'lsigma2', 'Gvar', 'Mvar', 'train_term1', 'train_term2', 'test_term1', 'test_term2']
    print('Full VI')
    print('{:<6s} {:<6s} {:<6s} {:<6s} {:<6s}  {:<6s} {:<6s} {:<10s} {:<10s} {:<10s} {:<10s}'.format(*header))
    for m in res:
        print_diagnostics(m)
    #
    # print('IP')
    # print('{:<5s} {:<5s} {:<5s} {:<5s}  {:<6s} {:<6s} {:<6s} {:<6s}'.format(*header))
    # for m in res_sp:
    #     print_diagnostics(m)