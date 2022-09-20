import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])
import glob

res_dir = r'code/test_results/surmise_MVGP_MVIP'
all_files = glob.glob(res_dir + "/*.csv")

li = []

for filename in all_files:
    row = pd.read_csv(filename, index_col=None, header=0)
    li.append(row)

df = pd.concat(li, axis=0, ignore_index=True)
df['frac'] = (df.n / df.p).astype(int)
methodlist = [('surmise', 1),
          ('MVGP', 1),
          ('MVIP', 1),
          ('MVIP', 2),
          ('MVIP', 4),
          ('MVIP', 8),
          ]
plt.figure()
for method, frac in methodlist:
    df0 = df.loc[(df.method==method) & (df.frac==frac)]
    plt.scatter(df0.n, df0.timeconstruct,
                label=r'{:s} $n={:d}p$'.format(method, frac), alpha=0.75)
plt.xlabel(r'$n$')
plt.ylabel(r'Construction time')
plt.yscale('log')
plt.legend()
plt.tight_layout()

plt.figure()
for method, frac in methodlist:
    df0 = df.loc[(df.method==method) & (df.frac==frac)]
    plt.scatter(df0.n, df0.testrmse,
                label=r'{:s} $n={:d}p$'.format(method, frac), alpha=0.75)
plt.xlabel(r'$n$')
plt.ylabel(r'RMSE')
plt.yscale('log')
plt.legend()
plt.tight_layout()

plt.figure()
for method, frac in methodlist:
    df0 = df.loc[(df.method==method) & (df.frac==frac)]
    plt.scatter(df0.n, df0.chi2,
                label=r'{:s} $n={:d}p$'.format(method, frac), alpha=0.75)
plt.xlabel(r'$n$')
plt.ylabel(r'$\chi^2$')
plt.yscale('log')
plt.legend()
plt.tight_layout()
# plt.scatter(df.n, )
