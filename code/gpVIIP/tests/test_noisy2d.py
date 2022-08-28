from func2d import forrester2008
import numpy as np
import torch
import matplotlib.pyplot as plt
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from optim_elbo import optim_elbo, optim_elbo_lbfgs
plt.style.use(['science', 'grid'])

x = np.random.uniform(0, 1, 25)
x[:3] = np.array((0.05, 0.85, 1.0))
# x = np.linspace(0.2, 1, 20)
f = forrester2008(x, noisy=True)

xtest = np.linspace(0, 1, 100)
ftest = forrester2008(xtest, noisy=False)


x = torch.tensor(x).unsqueeze(1)
f = torch.tensor(f)

xtest = torch.tensor(xtest).unsqueeze(1)
ftest = torch.tensor(ftest)

##############################################
# surmise block


from surmise.emulation import emulator

emu = emulator(f=f.numpy(), theta=x.numpy(),
               x=np.atleast_2d(np.array((1, 2))).T,
               method='PCGPwM',
               args={'warnings': True,
                     'epsilonPC': 25},
               options={})
emupred = emu.predict(theta=xtest.numpy(), x=np.atleast_2d(np.array((1, 2))).T)
emumean = emupred.mean()
emustd = np.sqrt(emupred.var())

emupct = emu._info['pct']

##############################################
model = MVN_elbo_autolatent(F=f, theta=x, kap=1, clamping=True)

print('train mse: {:.3E}, test mse: {:.3E}'.format(model.test_mse(theta0=x, f0=f),
                                                   model.test_mse(theta0=xtest, f0=ftest)))

model, niter, flag = optim_elbo_lbfgs(model, maxiter=200,
                                      lr=1e-1, gtol=1e-4,
                                      thetate=xtest, fte=ftest)

print('after training\ntrain mse: {:.3E}, '
      'test mse: {:.3E}'.format(model.test_mse(theta0=x, f0=f),
                                model.test_mse(theta0=xtest, f0=ftest)))

print(model.lLmb.grad, model.lsigma2.grad)

model.eval()
predmean = model.predictmean(xtest)
predstd = model.predictvar(xtest).sqrt()

def plot():
    lw = 3
    ymax = np.max((np.max(-(emupct.T @ (emumean + 2*emustd))),
                   np.max(((model.Phi / model.pcw).T @ (predmean + 2*predstd)).numpy())))
    ymin = np.min((np.min(-(emupct.T @ (emumean - 2*emustd))), np.min(((model.Phi / model.pcw).T @ (predmean - 2*predstd)).numpy())))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(xtest, -(emupct.T @ ftest.numpy()).T, linewidth=lw, label='True', color='k')
    ax[0].scatter(x, -(emupct.T @ f.numpy()).T, label='Data', color='gray')
    ax[0].plot(xtest, -(emupct.T @ emumean).T, linewidth=lw, label='Prediction', color='r')
    ax[0].plot(xtest, -(emupct.T @ (emumean + 2*emustd)).T, linewidth=lw, linestyle='--', color='r')
    ax[0].plot(xtest, -(emupct.T @ (emumean - 2*emustd)).T, linewidth=lw, linestyle='--', color='r')
    ax[0].set_title('PCGP', fontsize=30)

    ax[1].plot(xtest, ((model.Phi / model.pcw).T @ ftest).T, linewidth=lw, label='True', color='k')
    ax[1].scatter(x, ((model.Phi / model.pcw).T @ f).T, label='Data', color='gray')
    ax[1].plot(xtest, ((model.Phi / model.pcw).T @ predmean).T, linewidth=lw, label='Prediction', color='r')
    ax[1].plot(xtest, ((model.Phi / model.pcw).T @ (predmean + 2*predstd)).T, linewidth=lw, linestyle='--', color='r')
    ax[1].plot(xtest, ((model.Phi / model.pcw).T @ (predmean - 2*predstd)).T, linewidth=lw, linestyle='--', color='r')
    ax[1].set_title('PCGP w/ VI', fontsize=30)

    for axi in ax.flatten():
        axi.set(ylim=(ymin, ymax))
        axi.set_xlabel(r'$\boldsymbol x$', fontsize=24)
        axi.set_ylabel(r'$g(\boldsymbol x)$', fontsize=24, rotation=0,
                       labelpad=10)
        axi.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.legend(fontsize=24)
    plt.savefig('compare.png', dpi=300)
    plt.close()