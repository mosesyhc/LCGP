import matplotlib.pyplot as plt
plt.style.use(['science', 'no-latex', 'grid'])
plt.rcParams.update({'font.size': 14,
                     'lines.markersize': 12})
import torch
import numpy as np

from tests.func3d import forrester2008

noise = 1

n = 100
x = np.linspace(0, 1, n)
xpred = x[1:] - 1/(2*n)

ytrain = forrester2008(x, noisy=True, noises=[0.005, 0.1, 0.3])
truey = forrester2008(xpred, noisy=False)
newy = forrester2008(xpred, noisy=True, noises=[0.005, 0.1, 0.3])

x = torch.tensor(x).unsqueeze(1)
xpred = torch.tensor(xpred).unsqueeze(1)
ytrain = torch.tensor(ytrain)

# LCGP

from lcgp import LCGP

model = LCGP(y=ytrain, x=x, q=3) # , parameter_clamp=False)
model.compute_aux_predictive_quantities()
model.fit(verbose=False)

yhat, ypredvar, yconfvar, yfullcov = model.predict(xpred, return_fullcov=True)

# HetMOGP

from likelihoods.gaussian import Gaussian
from hetmogp.het_likelihood import HetLikelihood
from hetmogp.svmogp import SVMOGP
from hetmogp import util
from hetmogp.util import vem_algorithm as VEM

likelihood = HetLikelihood([Gaussian(), Gaussian(), Gaussian()])

M = int(0.25 * n)
Q = 3

ls_q = np.array(([.05]*Q))
var_q = np.array(([.5]*Q))
kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=1)

Z = np.linspace(0, 1, M)
Z = Z[:, np.newaxis]

model = SVMOGP(X=x.numpy(), Y=ytrain.numpy(), Z=Z, kern_list=kern_list, likelihood=likelihood, Y_metadata=None)
model = VEM(model)

# evaluation

import emulator_evaluation
emulator_evaluation.dss(newy, yhat.numpy(), yfullcov.numpy(), use_diag=False)
emulator_evaluation.intervalstats(newy, yhat.numpy(), ypredvar.numpy())

fig, ax = plt.subplots(1, 2, figsize=(12, 5)) #, sharey='row')
for j in range(model.q):
    ax[0].scatter(x, model.g.detach()[j], marker='.', label=noise, alpha=0.5)
    ax[0].set_ylabel('$g(x)$')
    ax[0].set_xlabel('$x$')
    # ax[0].plot(x, model.ghat.detach()[j],  label=noise, color='C{:d}'.format(j))
ax[0].legend(labels=['$g_1$', '$g_2$', '$g_3$'])

for j in range(model.p):
    ax[1].plot(xpred, truey[j], label=noise, color='k', linewidth=2)
    ax[1].set_ylabel('$f(x)$')
    ax[1].set_xlabel('$x$')
# ax[1].legend(labels=['$f_1$', '$f_2$', '$f_3$'])
    ax[1].scatter(x, f[j], marker='.', alpha=0.2, color='C{:d}'.format(j))
    ax[1].plot(xpred, yhat.detach()[j], label=noise, color='C{:d}'.format(j))
    ax[1].fill_between(xpred.squeeze(), (yhat - 2*yconfvar.sqrt()).detach()[j], (yhat + 2*yconfvar.sqrt()).detach()[j], alpha=0.5, color='C{:d}'.format(j))
    ax[1].fill_between(xpred.squeeze(), (yhat - 2*ypredvar.sqrt()).detach()[j], (yhat + 2*ypredvar.sqrt()).detach()[j], alpha=0.15, color='C{:d}'.format(j))
    ax[1].set_ylabel('$\hat{f}(x)$')
    ax[1].set_xlabel('$x$')
plt.tight_layout()

