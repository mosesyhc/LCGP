import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])
import glob

res_dir = r'C:\Users\cmyh\Documents\git\VIGP\code\test_results\surmise_MVGP_MVIP\save_VIIP'
all_files = glob.glob(res_dir + "/*.csv")

li = []

for filename in all_files:
    row = pd.read_csv(filename, index_col=None, header=0)
    li.append(row)

df = pd.concat(li, axis=0, ignore_index=True)
df['iipfrac'] = (df.n / df.p).astype(int).astype(str)

df['group'] = df.method+df.iipfrac

ylabels = {
           #  'testrmse': 'Test RMSE',
           # 'trainrmse': 'Train RMSE',
           # 'timeconstruct': 'Construction Time',
           # 'dss': 'DS Score',
           'crps': 'CRPS',
           # 'ccover': '95\% Coverage (confidence)',
           # 'pcover': '95\% Coverage (predictive)',
           # 'cintwid': '95\% Confidence Interval Width',
           # 'pintwid': '95\% Predictive Interval Width'
           }


for y, label in ylabels.items():
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 15
    sns.lineplot(x='n', y=y, hue='group', style='group',
                 markers=True, markersize=30,
                 linewidth=2.5, data=df)
    plt.ylabel(label)
    plt.xlabel(r'$N$')
    plt.xscale('log')
    if y != 'ccover' and y != 'pcover':
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(r'{:s}.png'.format(y), dpi=75)

    plt.close()