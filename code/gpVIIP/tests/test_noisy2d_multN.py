from func2d import forrester2008
import numpy as np
import torch
import matplotlib.pyplot as plt
from mvn_elbo_autolatent_model import MVN_elbo_autolatent
from optim_elbo import optim_elbo, optim_elbo_lbfgs
from surmise.emulation import emulator
import time
plt.style.use(['science', 'grid'])



def rmse(f, fhat):
    return np.sqrt(((f - fhat)**2).mean())


def crps(f, mu, sigma):
    from scipy.stats import norm
    z = (f - mu) / sigma

    pz = norm.pdf(z)
    cz = norm.cdf(z)

    s = -sigma * (1/np.sqrt(np.pi) - 2*pz - z *(2*cz - 1))
    return s.mean()

def ls(f, mu, sigma):
    from scipy.stats import norm
    pz = norm.pdf(f, mu, sigma)
    return -pz.mean()

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
    plt.savefig('compare{:d}.png'.format(x.shape[0]), dpi=300)
    plt.close()


if __name__ == '__main__':
    xtest = np.linspace(0, 1, 100)
    ftest = forrester2008(xtest, noisy=False)

    xtest = torch.tensor(xtest).unsqueeze(1)
    ftest = torch.tensor(ftest)

    Ns = [25, 50, 100, 250, 500, 1000]
    emuresult = np.zeros((len(Ns), 6))
    viresult = np.zeros((len(Ns), 6))
    for i, n in enumerate(Ns):
        x = np.random.uniform(0, 1, n)
        x[:3] = np.array((0.05, 0.85, 1.0))
        # x = np.linspace(0.2, 1, 20)
        f = forrester2008(x, noisy=True)

        x = torch.tensor(x).unsqueeze(1)
        f = torch.tensor(f)

        ##############################################
        # surmise block
        emu0 = time.time()
        emu = emulator(f=f.numpy(), theta=x.numpy(),
                       x=np.atleast_2d(np.array((1, 2))).T,
                       method='PCGPwM',
                       args={'warnings': True},
                       options={})
        emupred = emu.predict(theta=xtest.numpy(), x=np.atleast_2d(np.array((1, 2))).T)
        emumean = emupred.mean()
        emustd = np.sqrt(emupred.var())
        emu1 = time.time()

        emupct = emu._info['pct']

        ##############################################
        vi0 = time.time()
        model = MVN_elbo_autolatent(F=f, x=x, kap=1, clamping=True)

        print('train mse: {:.3E}, test mse: {:.3E}'.format(model.test_mse(x0=x, f0=f),
                                                           model.test_mse(x0=xtest, f0=ftest)))

        model, niter, flag = optim_elbo_lbfgs(model, maxiter=200,
                                              lr=1e-1, gtol=1e-4,
                                              thetate=xtest, fte=ftest)
        vi1 = time.time()

        print(model.lLmb.grad, model.lsigma2.grad)

        model.eval()
        predmean = model.predictmean(xtest)
        predstd = model.predictvar(xtest).sqrt()
        print('after training\ntrain mse: {:.3E}, '
              'test mse: {:.3E}'.format(model.test_mse(x0=x, f0=f),
                                        model.test_mse(x0=xtest, f0=ftest)))

        emuresult[i, :] = np.array((n, 0, emu1 - emu0, rmse(ftest.numpy(), emumean),
                                 crps(ftest.numpy(), emumean, emustd),
                                 ls(ftest.numpy(), emumean, emustd)))
        viresult[i, :] = np.array((n, 1, vi1 - vi0, rmse(ftest.numpy(), predmean.numpy()),
                                 crps(ftest.numpy(), predmean.numpy(), predstd.numpy()),
                                 ls(ftest.numpy(), predmean.numpy(), predstd.numpy())))
    np.savetxt(fname='emuresult.txt', X=emuresult)
    np.savetxt(fname='viresult.txt', X=viresult)
