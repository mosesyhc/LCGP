from mvn_elbo_autolatent_model import MVN_elbo_autolatent
import torch
import numpy as np
torch.set_default_dtype(torch.double)

def read_data(dir):
    f = np.loadtxt(dir + r'f.txt')
    x = np.loadtxt(dir + r'x.txt')
    theta = np.loadtxt(dir + r'theta.txt')
    return f, x, theta

fname = r'\wingweight_data\\'
dir = r'code\data' + fname
f0, x0, xtr = read_data(dir)


n = 100
m = 50
f1 = f0[1, :n]
f2 = f0[0, :n]

f = np.zeros((m, n))
for i in range(int(m / 2)):
    f[2*i] = f1 + np.random.normal(0, 0.1 * f1.std(), n)
    f[2*i + 1] = f2 + np.random.normal(0, 0.1 * f2.std(), n)

f = torch.tensor(f)
x = torch.tensor(xtr[:n])
#
# models = []
# lsigma2s = []
# lsigma2start = []
# lmse0s = []
# kaps = torch.tensor((1, 2, 5, 10, 20, 50))
# for kap in kaps:
#     model = MVN_elbo_autolatent(F=f, x=x, kap=kap, clamping=False)
#     # print(model.lsigma2)
#     # model.fit(verbose=True)
#     # models.append(model)
#     lmse0s.append(model.lmse0)
#     lsigma2start.append(model.lsigma2)
#     # lsigma2s.append(model.lsigma2)
# # lsigma2s = torch.tensor(lsigma2s)
# lsigma2start = torch.tensor(lsigma2start)
# #
model = MVN_elbo_autolatent(F=f, x=x, kap=2, clamping=False)

# model.fit_adam(verbose=True)
model.fit(verbose=True)

negelbos = []
lsigma2s = torch.arange(-6, 0, 0.01)
for lsigma2 in lsigma2s:
    model.set_lsigma2(lsigma2)
    negelbos.append(model.negelbo())

negelbos = torch.tensor(negelbos)
import matplotlib.pyplot as plt

plt.plot(lsigma2s, negelbos)

import matplotlib.pyplot as plt
#
# # plt.plot()
#
# from surmise.emulation import emulator
# emu2 = emulator(x=np.arange(50), theta=x.numpy(), f=f.numpy(),
#                method='PCGPwM', args={'epsilonPC':5})