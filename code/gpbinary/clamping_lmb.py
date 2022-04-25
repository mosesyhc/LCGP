import torch
import matplotlib.pyplot as plt

def clam_lmb(x, c):
    return 2.5 * torch.tanh(c*(x))

x = torch.arange(-5, 5, 0.01)
# mindiff = torch.inf
# i0 = torch.inf
for i in torch.arange(0.001, 1, 0.001):
    plt.plot(x, clam_lmb(x, i), alpha=0.1, color='grey')
    # diff = ((x - clam_lmb(x, i)) ** 2).mean()
    # print('x: {:.6f}, diff: {:.6f}'.format(i, diff))
    # if diff < mindiff:
    #     mindiff = diff
    #     i0 = i

plt.plot(x, x, linewidth=2, color='k')
plt.plot(x, clam_lmb(x, 0.4925), linewidth=2, color='r', label=r'c = {:.4f}'.format(0.4925))
plt.vlines((-2.3, 2.3), -5, 5, linestyle='dashed')
plt.title(r'$2.5 * tanh(cx)$')
plt.legend()
plt.tight_layout()
plt.savefig(r'clamp_lmb.png', dpi=150)

