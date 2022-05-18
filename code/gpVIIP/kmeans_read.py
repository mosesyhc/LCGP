import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['science', 'no-latex', 'grid'])

df = pd.read_csv(r'code\test_results\kmeans_trial.csv')
df['lsigma2'] = df['lsigma2'].astype(str)
# df['label'] = 'n = ' + (1/df['ip_frac']).astype(int).astype(str) + 'p'
fig1, ax_meanD = plt.subplots(2, 2, sharey=True, figsize=(8, 6.5))
axes_meanD = ax_meanD.flatten()
for i, ip_frac in enumerate(df['ip_frac'].unique()):
    ddf = df[df['ip_frac'] == ip_frac]
    sns.lineplot(x='n', y='d_mean_I', hue='lsigma2', data=ddf, ax=axes_meanD[i])
    axes_meanD[i].set_title('IP fraction = 1/{:d}'.format(int(1/ip_frac)))
plt.tight_layout()
plt.show()

fig2, ax_maxD = plt.subplots(2, 2, sharey=True, figsize=(8, 6.5))
axes_maxD = ax_maxD.flatten()
for i, ip_frac in enumerate(df['ip_frac'].unique()):
    ddf = df[df['ip_frac'] == ip_frac]
    sns.lineplot(x='n', y='d_max_I', hue='lsigma2', data=ddf, ax=axes_maxD[i])
    axes_maxD[i].set_title('IP fraction = 1/{:d}'.format(int(1/ip_frac)))
    # axes_maxD[i].set_yscale('log')
plt.tight_layout()
plt.show()
