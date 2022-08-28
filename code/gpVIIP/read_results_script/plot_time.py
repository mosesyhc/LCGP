import glob
import pandas as pd

res_dir = r'C:\Users\cmyh\Documents\git\binary-hd-emulator\code\test_results\borehole_comparisons\comparison_20220515_jit_detach2'
all_files = glob.glob(res_dir + "/*.csv")

li = []

for filename in all_files:
    row = pd.read_csv(filename, index_col=None, header=0)
    li.append(row)

import torch
df = pd.concat(li, axis=0, ignore_index=True)


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['science', 'no-latex', 'grid'])
plt.figure(figsize=(5,5))
sns.lineplot(x='n', y='timeconstruct', linewidth=2.5,
             hue='label',
             style='label', data=df)
plt.ylabel('build time')
# plt.yscale('log')
# plt.xscale('log')
plt.legend(title='')
plt.tight_layout()