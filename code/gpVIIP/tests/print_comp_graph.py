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

    # emu, emumean, emustd, emupcto, \
    #     n, xtr, ftr, xte, fte = test_single(method='surmise', n=n, seed=0, noiseconst=noiseconst,
    #                                         ftr=f, xtr=x, fte=ftest, xte=xtest,
    #                                         fte0=ftest0, output_csv=True, dir=DIR,
    #                                         return_quant=True)

    model, predmean, predstd, Phi, \
        _, _, _, _, _ = test_single(method='MVGP', n=n, seed=0, # kap=1,
                                    noiseconst=noiseconst,
                                    ftr=f, xtr=x, fte=ftest, xte=xtest,
                                    fte0=ftest0, output_csv=True, dir=DIR,
                                    return_quant=True)

    # model, predmean, predstd, Phi, \
    #     _, _, _, _, _ = test_single(method='MVIP', n=n, ip_frac=1/4, seed=0, # kap=1,
    #                                 noiseconst=noiseconst,
    #                                 ftr=f, xtr=x, fte=ftest, xte=xtest,
    #                                 fte0=ftest0, output_csv=True, dir=DIR,
    #                                 return_quant=True)

    return model  #, modelsp

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
    #
    # def print_diagnostics(m):
    #     lLmb, lsigma2, lnug, ltau = m.parameter_clamp(m.lLmb, m.lsigma2, m.lnugGPs, m.ltau2GPs)
    #     testGPvar1, testGPvar2 = m.GPvarterm1.mean().item(), m.GPvarterm2.mean().item()
    #     _ = m.predictcov(m.theta)
    #     trainGPvar1, trainGPvar2 = m.GPvarterm1.mean().item(), m.GPvarterm2.mean().item()
    #     Gvar = m.G.var()
    #     Mvar = m.M.var()
    #     print('{:<6d} {:<6.3f} {:<6.3f} {:<6.3f} {:<6.3f} {:<6.3f} {:<6.3f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
    #         m.n, ltau[0].item(), lnug[0].item(), lLmb[0, -1].item(), lsigma2.item(), Gvar, Mvar, trainGPvar1, trainGPvar2, testGPvar1, testGPvar2))
    #
    # header = ['n', 'ltau2gps', 'lnug', 'llmb[-1]', 'lsigma2', 'Gvar', 'Mvar', 'train_term1', 'train_term2', 'test_term1', 'test_term2']
    # print('Full VI')
    # print('{:<6s} {:<6s} {:<6s} {:<6s} {:<6s}  {:<6s} {:<6s} {:<10s} {:<10s} {:<10s} {:<10s}'.format(*header))
    # for m in res:
    #     print_diagnostics(m)
    #
    # print('IP')
    # print('{:<5s} {:<5s} {:<5s} {:<5s}  {:<6s} {:<6s} {:<6s} {:<6s}'.format(*header))
    # for m in res_sp:
    #     print_diagnostics(m)