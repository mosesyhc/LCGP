import torch
import matplotlib.pyplot as plt

def clam_lsigma2(x, c):
    return -13/2 + 11/2 * torch.tanh(c*(x + 13/2))

x = torch.arange(-18, 5, 0.01)
# mindiff = torch.inf
# i0 = torch.inf
for i in torch.arange(0.05, 0.45, 0.001):
    plt.plot(x, clam_lsigma2(x, i), alpha=0.15, color='grey')
    diff = ((x - clam_lsigma2(x, i)) ** 2).mean()
    # print('x: {:.6f}, diff: {:.6f}'.format(i, diff))
    # if diff < mindiff:
    #     mindiff = diff
    #     i0 = i

plt.plot(x, x, linewidth=2, color='k')
plt.plot(x, clam_lsigma2(x, 0.224), linewidth=2, color='r', label=r'c = {:.3f}'.format(0.224))
plt.vlines((-12, -1), -18, 5, linestyle='dashed')
plt.title(r'$-13/2 + 11/2 * tanh(c * (x + 13/2))$')
plt.legend()
plt.tight_layout()
plt.savefig(r'clamp_lsigma2.png', dpi=150)

