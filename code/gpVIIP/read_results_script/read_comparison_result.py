import pandas as pd
import glob, os

res_dir = r'code\test_results\borehole_comparisons\comparison_20220515_jit_detach'
all_files = glob.glob(res_dir + "/*.csv")

li = []

for filename in all_files:
    row = pd.read_csv(filename, index_col=None, header=0)
    li.append(row)

import torch
df = pd.concat(li, axis=0, ignore_index=True)
# df['trainrmse'] = df['trainrmse'].astype(str)
# df['testrmse'] = df['testrmse'].astype(str)
# df['trainrmse'] = df['trainrmse'].str.strip('tensor')
# df['trainrmse'] = df['trainrmse'].str.strip('(')
# df['trainrmse'] = df['trainrmse'].str.strip(')')
# df['testrmse'] = df['testrmse'].str.strip('tensor')
# df['testrmse'] = df['testrmse'].str.strip('(')
# df['testrmse'] = df['testrmse'].str.strip(')')
# df['trainrmse'] = df['trainrmse'].astype(float)
# df['testrmse'] = df['testrmse'].astype(float)
df['ip_frac_inv'] = df.n // df.p
df['label'] = df.method
df['label'][df.method == 'MVIP'] = df.method[df.method == 'MVIP'] + ' n=' + df['ip_frac_inv'][df.method == 'MVIP'].astype(str) + 'p'

order = ['surmise', 'MVGP', 'MVIP n=1p', 'MVIP n=2p'] #, 'MVIP n=4p'] #, 'MVIP n=8p']
dfwo = df[df.method != 'surmise']
orderwo = ['MVGP', 'MVIP n=1p', 'MVIP n=2p', 'MVIP n=4p', 'MVIP n=8p']
labelwo = ['no IP', r'$n=1p$', r'$n=2p$', r'$n=4p$', r'$n=8p$']

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['science', 'grid'])
plt.figure(figsize=(8, 5))
sns.lineplot(x='n', y='timeconstruct', linewidth=4,
             ci=None,
             hue='label', hue_order=orderwo,
             style='label', style_order=orderwo, data=dfwo)
plt.ylabel(r'Construction time, $s$', fontsize=24, labelpad=12)
plt.xlabel(r'$N$', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.yscale('log')
# plt.xscale('log')
plt.legend(title='', labels=labelwo, fontsize=18)
plt.tight_layout()
plt.savefig('time.png', dpi=300)
plt.close()

plt.figure(figsize=(5,5))
sns.lineplot(x='n', y='trainrmse', linewidth=2.5,
             hue='label', hue_order=order,
             style='label', style_order=order, data=df)
# plt.yscale('log')
# plt.xscale('log')
plt.ylabel('train rmse')
plt.legend(title='')
plt.tight_layout()


plt.figure(figsize=(5,5))
sns.lineplot(x='n', y='testrmse', linewidth=2.5,
             hue='label', hue_order=order,
             style='label', style_order=order, data=df)
# plt.yscale('log')
# plt.xscale('log')
plt.ylabel('test rmse')
plt.legend(title='')
plt.tight_layout()

# sns.boxplot(data=df, x='label', y='testrmse', hue='label', hue_order=order)