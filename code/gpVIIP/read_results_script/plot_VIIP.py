import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])
import glob

res_dir = r'code\test_results\surmise_MVGP_MVIP\param05'
out_dir = r'code\fig\talk-figs'
all_files = glob.glob(res_dir + "/*.csv")

li = []

for filename in all_files:
    row = pd.read_csv(filename, index_col=None, header=0)
    li.append(row)

df = pd.concat(li, axis=0, ignore_index=True)
df['iipfrac'] = (df.n / df.p).astype(int).astype(str)
df = df.loc[df.method != 'surmise']

df['group'] = df.method+df.iipfrac

ylabels = {
           'testrmse': 'Test RMSE',
           'trainrmse': 'Train RMSE',
           'timeconstruct': 'Construction Time',
           'dss': 'DS Score',
           'crps': 'CRPS',
           'ccover': '95\% Coverage (confidence)',
           'pcover': '95\% Coverage (predictive)',
           'cintwid': '95\% Confidence Interval Width',
           'pintwid': '95\% Predictive Interval Width'
           }
#
#
# for y, label in ylabels.items():
#     plt.figure(figsize=(8, 6))
#     plt.rcParams["font.size"] = 15
#     sns.lineplot(x='n', y=y, hue='group', style='group',
#                  markers=True, markersize=30,
#                  linewidth=2.5, data=df)
#     plt.ylabel(label)
#     plt.xlabel(r'$N$')
#     plt.xscale('log')
#     if y != 'ccover' and y != 'pcover':
#         plt.yscale('log')
#     plt.tight_layout()
#     plt.savefig(out_dir + r'\{:s}.png'.format(y), dpi=75)
#
#     plt.close()



import matplotlib.pyplot as plt
plt.style.use(['science', 'high-contrast', 'grid'])
plt.rcParams.update({'font.size': 14,
                     'lines.markersize': 12})

gpcolor = r'#7b3294'
gpshade = r'#c2a5cf'
vicolor = r'#008837'
vishade = r'#a6dba0'

plt.figure(figsize=(4, 5))
sns.lineplot(x='n', y='timeconstruct',
             hue='group', style='group',
             linewidth=4, data=df)
plt.yscale('log')
plt.xlabel(r'$n$')
plt.ylabel('Construction Time')

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 0, 2]
labels = [r'VI, $n = 4m$', r'VI, w/o IP', r'VI, $n = 8m$']
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
plt.tight_layout()

