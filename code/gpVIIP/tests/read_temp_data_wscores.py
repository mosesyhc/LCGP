import pandas as pd
import numpy as np
import glob

def median(a):
    return(np.median(a))

res_dir = r'C:\Users\moses\Desktop\git\binary-hd-emulator\code\test_results\surmise_MVGP_MVIP\2dExample_noise'

all_files = glob.glob(res_dir + "/*.csv")

li = []

for filename in all_files:
    row = pd.read_csv(filename, index_col=None, header=0)
    li.append(row)

df = pd.concat(li, axis=0, ignore_index=True)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['science', 'grid'])
plt.rcParams["font.size"] = 10

ys = ['trainrmse', 'testrmse',
      'crps', 'dss',
      'pcover', 'pintscore']
ylabs = ['Train RMSE', 'Test RMSE',
         'CRPS', 'DS Score',
         'Predictive Coverage', 'Predictive Interval Score']

order = ['surmise', 'MVIP', 'MVGP']

k = 0
fig, axes = plt.subplots(3, 2, figsize=(9, 12))
for ax in axes.flatten():
    sns.lineplot(x='noiseconst', y=ys[k],
                 hue='method', style='method',
                 hue_order=order, style_order=order,
                 # estimator=median,
                 ci=None,
                 markers=True, markersize=15,
                 linewidth=2.5, data=df, ax=ax)
    ax.set_xlabel(r'Noise stdev multiplier')
    ax.set_ylabel(ylabs[k])
    if ys[k] != 'pcover':
        ax.set_yscale('log')
    # ax.set_xscale('log')
    k += 1
plt.tight_layout()
plt.savefig('compile_all.png', dpi=75)
plt.close()

k = 0
fig, axes = plt.subplots(3, 2, figsize=(9, 12))
for ax in axes.flatten():
    sns.lineplot(x='noiseconst', y=ys[k],
                 hue='method', style='method',
                 hue_order=order[:-1], style_order=order[:-1],
                 # estimator=median,
                 ci=None,
                 markers=True, markersize=12,
                 linewidth=2.5, data=df.loc[df.method!='MVGP'], ax=ax)
    ax.set_xlabel(r'Noise stdev multiplier')
    ax.set_ylabel(ylabs[k])
    if ys[k] not in ['pcover', 'CRPS']:
        ax.set_yscale('log')
    # ax.set_xscale('symlog')
    k += 1
plt.tight_layout()
plt.savefig('compile_notfull.png', dpi=75)
plt.close()
# # plt.figure(figsize=(4, 3))
# sns.lineplot(x='noiseconst', y='trainrmse',
#              hue='method', style='method',
#              markers=True, markersize=10,
#              linewidth=2.5, data=df)
# plt.xlabel('Noise stdev multiplier')
# plt.ylabel('Train RMSE')
# # plt.savefig('trainrmse.png', dpi=75)
# # plt.close()
#
#
# # plt.figure(figsize=(4, 3))
# sns.lineplot(x='noiseconst', y='testrmse',
#              hue='method', style='method',
#              markers=True, markersize=10,
#              linewidth=2.5, data=df)
# plt.xlabel('Noise stdev multiplier')
# plt.ylabel('Test RMSE')
# # plt.savefig('testrmse.png', dpi=75)
# # plt.close()
#
#
# # plt.figure(figsize=(4, 3))
# sns.lineplot(x='noiseconst', y='crps',
#              hue='method', style='method',
#              markers=True, markersize=10,
#              linewidth=2.5, data=df)
# plt.xlabel('Noise stdev multiplier')
# plt.ylabel('CRPS')
# # plt.savefig('crps.png', dpi=75)
# # plt.close()
#
#
#
# # plt.figure(figsize=(4, 3))
# sns.lineplot(x='noiseconst', y='dss',
#              hue='method', style='method',
#              markers=True, markersize=10,
#              linewidth=2.5, data=df)
# plt.xlabel('Noise stdev multiplier')
# plt.ylabel('DS Score')
# # plt.savefig('dss.png', dpi=75)
# # plt.close()
#
#
#
# # plt.figure(figsize=(4, 3))
# sns.lineplot(x='noiseconst', y='pcover',
#              hue='method', style='method',
#              markers=True, markersize=10,
#              linewidth=2.5, data=df)
# plt.xlabel('Noise stdev multiplier')
# plt.ylabel('Predictive coverage')
# # plt.savefig('pcover.png', dpi=75)
# # plt.close()
#
#
# # plt.figure(figsize=(4, 3))
# sns.lineplot(x='noiseconst', y='pintwid',
#              hue='method', style='method',
#              markers=True, markersize=10,
#              linewidth=2.5, data=df)
# plt.xlabel('Noise stdev multiplier')
# plt.ylabel('Predictive interval width')
# # plt.savefig('pintwid.png', dpi=75)
# # plt.close()
#
# # plt.figure(figsize=(4, 3))
# sns.lineplot(x='noiseconst', y='pintscore',
#              hue='method', style='method',
#              markers=True, markersize=10,
#              linewidth=2.5, data=df))
# plt.xlabel('Noise stdev multiplier')
# plt.ylabel('Predictive interval score')
# # plt.savefig('pintscore.png', dpi=75)
# # plt.close()
