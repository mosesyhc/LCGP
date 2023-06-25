import surmise
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_dir = r'code/data/colin_data/'
    theta = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_input.csv')
    f = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_output.csv')
    theta = theta.iloc[:, 1:].to_numpy()
    f = f.iloc[:, 1:].to_numpy().T

    # f = ((f.T - f.min(1).values) / (f.max(1).values - f.min(1).values)).T
    f = ((f.T - f.mean(1)) / f.std(1)).T

    # arbitrary x
    m, n_all = f.shape

    ntr = 200
    indtr = np.random.permutation(n_all)[:ntr]
    indte = np.setdiff1d(np.arange(n_all), indtr)[:105]
    ftr = f[:, indtr]
    thetatr = theta[indtr]
    fte = f[:, indte]
    thetate = theta[indte]

    kap = 2
    Phi, _, _ = np.linalg.svd(ftr, full_matrices=False)
    Phi = Phi[:, :kap]
    Phi_mse = ((ftr - Phi @ Phi.T @ ftr)**2).mean()
    print('recovery mse: {:.3f}'.format(Phi_mse))

    x = np.arange(kap).reshape(kap, 1)
    g = Phi.T @ ftr

    g[0, 100] = np.nan

    from surmise.emulation import emulator
    emu = emulator(x=x, theta=thetatr, f=g, method='PCGPwM', )
    emupred = emu.predict(x=x, theta=thetate)