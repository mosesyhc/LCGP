import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from test_general import test_single
from mvn_elbo_autolatent_model import MVN_elbo_autolatent


def read_test_data(dir):
    testf = np.loadtxt(dir + r'testf.txt')
    testtheta = np.loadtxt(dir + r'testtheta.txt')
    return testf, testtheta


def read_data(dir):
    f = np.loadtxt(dir + r'f.txt')
    x = np.loadtxt(dir + r'x.txt')
    theta = np.loadtxt(dir + r'theta.txt')
    return f, x, theta


fname = r'\wingweight_data\\'
dir = r'C:\Users\moses\Desktop\git\VIGP\code\data' + fname
f, x0, xtr = read_data(dir)
fte0, xte = read_test_data(dir)

m, ntr = f.shape
fstd = f.std(1)
ftr = np.zeros_like(f)

fte = np.zeros_like(fte0)
_, nte = fte.shape

noiseconst = 0.25
for j in range(m):
    ftr[j] = f[j] + np.random.normal(0, noiseconst * fstd[j], ntr)
    fte[j] = fte0[j] + np.random.normal(0, noiseconst * fstd[j], nte)

ftr = torch.tensor(ftr)
fte0 = torch.tensor(fte0)
fte = torch.tensor(fte)
xtr = torch.tensor(xtr)
xte = torch.tensor(xte)

n = 100
torch.manual_seed(0)
tr_ind = torch.randperm(ntr)[:n]
ftr_n = ftr[:, tr_ind]
xtr_n = xtr[tr_ind]
torch.seed()

model = MVN_elbo_autolatent(F=ftr_n, x=xtr_n,
                                    clamping=True)
pct = model.Phi
kap = model.kap
niter, flag = model.fit(sep=False, verbose=True)
predmeantr = model.predictmean(xtr_n).detach().numpy()
predmean = model.predictmean(xte).detach().numpy()
predcov = model.predictcov(xte).detach().numpy()

from surmise.emulation import emulator
emu = emulator(x=np.arange(ftr_n.shape[0]), theta=xtr_n.numpy(), f=ftr_n.numpy(),
               method='PCGPwM')