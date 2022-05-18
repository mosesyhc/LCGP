import pandas as pd
import glob, os

res_dir = r'code/test_results/comparison_20220515_jit_detach2'
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

order = ['surmise', 'MVGP', 'MVIP n=1p', 'MVIP n=2p', 'MVIP n=4p', 'MVIP n=8p']
dfwo = df[df.method != 'surmise']
orderwo = ['MVGP', 'MVIP n=1p', 'MVIP n=2p', 'MVIP n=4p', 'MVIP n=8p']

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['science', 'no-latex', 'grid'])
plt.figure(figsize=(5,5))
sns.lineplot(x='n', y='timeconstruct', linewidth=2.5,
             hue='label', hue_order=order,
             style='label', style_order=order, data=df)
plt.ylabel('build time')
plt.yscale('log')
plt.xscale('log')
plt.legend(title='')
plt.tight_layout()

plt.figure(figsize=(5,5))
sns.lineplot(x='n', y='trainrmse', linewidth=2.5,
             hue='label', hue_order=order,
             style='label', style_order=order, data=df)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('train rmse')
plt.legend(title='')
plt.tight_layout()


plt.figure(figsize=(5,5))
sns.lineplot(x='n', y='testrmse', linewidth=2.5,
             hue='label', hue_order=order,
             style='label', style_order=order, data=df)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('test rmse')
plt.legend(title='')
plt.tight_layout()
