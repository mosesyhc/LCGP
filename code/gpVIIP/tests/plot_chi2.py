import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])

df = pd.read_csv(r'noisy2d_chi2.csv')

plt.figure(figsize=(6, 6))
# plt.scatter(df.n, df.surmise, label='surmise')
plt.scatter(df.n, df.VI, label='VI')
plt.hlines(1, 0, max(df.n), linestyles='--', colors='k')
plt.legend()
plt.xlabel(r'$n$')
plt.ylabel(r'$(f - \hat{f})^2 / \sigma^2$')
