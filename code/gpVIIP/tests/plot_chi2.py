import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])

df = pd.read_csv(r'noisy2d_chi2.csv')

plt.figure(figsize=(6, 6))
plt.scatter(df.n, df.surmiseChi2, label='surmise', alpha=0.5)
plt.scatter(df.n, df.VIChi2, label='VI', alpha=0.5)
plt.hlines(1, 0, max(df.n), linestyles='--', colors='k')
plt.legend()
plt.yscale('log')
plt.xlabel(r'$n$')
plt.ylabel(r'$(f - \hat{f})^2 / \sigma^2$')
plt.tight_layout()


plt.figure(figsize=(6, 6))
plt.scatter(df.n, df.surmiseRMSE, label='surmise', alpha=0.5)
plt.scatter(df.n, df.VIRMSE, label='VI', alpha=0.5)
plt.legend()
plt.yscale('log')
plt.xlabel(r'$n$')
plt.ylabel(r'$\sqrt{\sum(f - \hat{f})^2}$')
plt.tight_layout()

#
# plt.figure()
# plt.scatter(df.n, pd.exp(df.lsigma2), alpha=0.5)
# plt.yscale('log')
# plt.xlabel(r'$n$')
# plt.ylabel(r'$\sigma^2$')