import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)

import numpy as np
import pandas as pd

from prediction import pred_gp
from likelihood import negloglik_gp


class simpleGP(nn.Module):
    def __init__(self, llmb, lsigma2, theta, g):
        super().__init__()
        self.llmb = nn.Parameter(llmb)
        self.lsigma2 = nn.Parameter(lsigma2)
        self.theta = theta
        self.g = g

    def forward(self, theta0):
        llmb = self.llmb
        lsigma2 = self.lsigma2
        theta = self.theta
        g = self.g

        pred, predvar = pred_gp(llmb, lsigma2, theta, theta0, g)
        return pred

    def lik(self):
        llmb = self.llmb
        lsigma2 = self.lsigma2
        theta = self.theta
        g = self.g
        return negloglik_gp(llmb, lsigma2, theta, g)

    def test_mse(self, theta0, g0):
        gpred, gpredvar = pred_gp(self.llmb, self.lsigma2, self.theta, theta0, self.g)
        return ((gpred - g0) ** 2).mean()


data_dir = r'./code/data/colin_data/'
theta = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_input.csv')
f = pd.read_csv(data_dir + r'ExpandedRanges2_LHS1L_n1000_s0304_all_output.csv')
theta = torch.tensor(theta.iloc[:, 1:].to_numpy())
f = torch.tensor(f.iloc[:, 1:].to_numpy()).T

f = ((f.T - f.mean(1)) / f.std(1)).T

# arbitrary x
m, n_all = f.shape
x = np.arange(m)

ftr = f[0, :400].unsqueeze(1)
thetatr = theta[:400]

fte = f[0, 500:700].unsqueeze(1)
thetate = theta[500:700]

llmb = torch.Tensor(0.5 * torch.log(torch.Tensor([thetatr.shape[1]])) +
                    torch.log(torch.std(thetatr, 0)))
lsigma2 = torch.var(ftr).log()

model = simpleGP(llmb, lsigma2, thetatr, ftr)
model.double()
model.requires_grad_()


optim = torch.optim.FullBatchLBFGS(model.parameters())
mse0 = model.test_mse(thetate, fte)

# closure
def closure():
    optim.zero_grad()
    return model.lik()

loss = closure()
loss.backward()

header = ['iter', 'negloglik', 'test mse']
print('{:<5s} {:<12s} {:<12s}'.format(*header))
print('{:<5s} {:<12.6f} {:<10.3f}'.format(' ', loss, mse0))
for epoch in range(25):
    options = {'closure': closure, 'current_loss': loss}
    loss, grad, lr, _, _, _, _, _ = optim.step(options)

    mse = model.test_mse(thetate, fte)
    print('{:<5d} {:<12.6f} {:<10.3f}'.format(epoch, loss, mse))

print(model.llmb.grad)
print(model.llmb)
print(model.lsigma2.grad)
print(model.lsigma2)

ftrpred, ftrpredvar = pred_gp(model.llmb, model.lsigma2, thetatr, thetatr, ftr)
chitr = (((ftrpred - ftr).squeeze() ** 2) / ftrpredvar).detach().numpy()
stderrtr = (((ftrpred - ftr).squeeze()) / ftrpredvar.sqrt()).detach().numpy()

print(chitr.mean())


fpred, fpredvar = pred_gp(model.llmb, model.lsigma2, thetatr, thetate, ftr)
chi = (((fpred - fte).squeeze() ** 2) / fpredvar).detach().numpy()
stderr = (((fpred - fte).squeeze()) / fpredvar.sqrt()).detach().numpy()

largechiind = chi > np.quantile(chi, 0.95)
smallchiind = chi <= np.quantile(chi, 0.95)

chi[smallchiind].mean()

import seaborn as sns
thetatrplot = pd.DataFrame(thetatr)
thetatrplot['error'] = 'train'
thetateplot = pd.DataFrame(thetate)
huetest = ['low']*thetate.shape[0]
huetest = np.array(huetest)
huetest[largechiind] = 'high'
thetateplot['error'] = huetest

thetaall = pd.concat((thetatrplot, thetateplot), ignore_index=True)

# p = sns.pairplot(thetaall, hue='error', markers=["o", "s", "D"],
#              palette=sns.set_palette(sns.color_palette(['#66c2a5', '#8da0cb', '#fc8d62'])), corner=True,
#              diag_kind='hist',
#              diag_kws={'common_norm': True})

import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(np.arange(150), stderr)
# plt.show()

plt.figure()
plt.hist(np.random.normal(0, 1, 1000), alpha=0.7, density=True, label='N(0,1)', bins=50)
plt.hist(stderrtr, alpha=0.7, density=True, label='standard error', bins=50)
plt.xlabel(r'$ (f - \mu) / \sigma$, Training')
plt.legend()
plt.show()

plt.figure()
plt.hist(np.random.normal(0, 1, 1000), alpha=0.7, density=True, label='N(0,1)', bins=30)
plt.hist(stderr, alpha=0.7, density=True, label='standard error', bins=50)
plt.xlabel(r'$ (f - \mu) / \sigma$, Testing')
plt.legend()
plt.show()