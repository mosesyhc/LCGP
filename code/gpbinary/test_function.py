import numpy as np
import torch

def borehole(theta):
    """
    Wraps the borehole function
    """
    rw = theta[0]
    r = theta[1]
    Tu = theta[2]
    Hu = theta[3]
    Tl = theta[4]
    Hl = theta[5]
    L = theta[6]
    Kw = theta[7]
    frac1 = 2 * np.pi * Tu * (Hu - Hl)
    frac2a = (2 * L * Tu) / (np.log(r / rw) * (rw ** 2) * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1 + frac2a + frac2b)
    f = frac1 / frac2
    return f


def gen_true_theta():
    """Generate one parameter to be the true parameter for calibration."""
    thetalimits = np.array([[0.05, 0.15], #rw
                            [100, 50000], # r
                            [63070, 115600], # Tu
                            [990, 1110], # Hu
                            [63.1, 116], # Tl
                            [700, 820], # Hl
                            [1120, 1680], # L
                            [9855, 12045]]) # Kw
    theta = np.random.uniform(thetalimits[:,0],
                              thetalimits[:,1])
    return theta


def gen_borehole_data(ntrain=20, ntest=50):
    theta = torch.zeros(ntrain, 8)
    for k in range(ntrain):
        theta[k] = torch.tensor(gen_true_theta())
    f = borehole(theta.T).unsqueeze(1)

    fmean = f.mean()
    fstd = f.std()
    f = (f - fmean) / fstd
    thmean = theta.mean(0)
    thstd = theta.std(0)
    theta = (theta - thmean) / thstd

    thetatest = torch.zeros(ntest, 8)
    for k in range(ntest):
        thetatest[k] = torch.tensor(gen_true_theta())
    ind0 = torch.argsort(thetatest[:, 1])
    thetatest = thetatest[ind0]
    ftest = borehole(thetatest.T).unsqueeze(1)
    ftest = (ftest - fmean) / fstd
    thetatest = (thetatest - thmean) / thstd

    return theta, f, thetatest, ftest
