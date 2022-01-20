import torch.distributions.normal as Normal
norm = Normal.Normal(0, 1)
import numpy as np


def read_data(dir):
    f = np.loadtxt(dir + r'f.txt')
    x = np.loadtxt(dir + r'x.txt')
    theta = np.loadtxt(dir + r'theta.txt')
    return f, x, theta


def read_only_complete_data(dir):
    f = np.loadtxt(dir + r'f.txt')
    x = np.loadtxt(dir + r'x.txt')
    theta = np.loadtxt(dir + r'theta.txt')

    comp_inds = np.isnan(f).sum(0) == 0
    f = f[:, comp_inds]
    theta = theta[comp_inds]

    return f, x, theta


def visualize_dataset(ytrain, ytest):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(-ytrain.T, aspect='auto', cmap='gray', interpolation='none')
    ax[1].imshow(-ytest.T, aspect='auto', cmap='gray', interpolation='none')
    ax[0].set_title('Training data')
    ax[0].set_ylabel('Parameters')
    ax[1].set_title('Testing data')
    plt.show()

def get_psi(y):
    # y = self.y
    z = (y.sum(1) + 10) / (y.shape[1] + 20)
    psi = norm.icdf(z)
    return psi.unsqueeze(1)  # returns m x 1


def get_Phi(x):
    # x = self.x
    tmp = x[:, :2]
    tmp[:, 0] -= tmp[:, 1]  # Use (N, Z) instead of (A, Z)
    Phi = (tmp - tmp.mean(0)) / tmp.std(0)
    return Phi  # returns m x kappa
