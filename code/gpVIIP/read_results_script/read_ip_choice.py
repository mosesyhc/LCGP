import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])
import glob

res_dir = r'code/test_results/choice_ip'
all_files = glob.glob(res_dir + "/*.csv")

li = []

for filename in all_files:
    row = pd.read_csv(filename, index_col=None, header=0)
    li.append(row)

df = pd.concat(li, axis=0, ignore_index=True)
df.dropna()
df = df[df['title'] != 'kmeans_w_art']

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
sns.lineplot(x='p', y='testrmse', hue='title', style='title',
             linewidth=2.5,
             markers=True, data=df)
plt.legend(title='')
plt.ylabel('Test RMSE')
plt.xlabel('Number of inducing points, p')
# plt.yscale('log')
plt.title('$n=400$')
plt.tight_layout()

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15
sns.lineplot(x='p', y='trainrmse', hue='title', style='title',
             linewidth=2.5,
             markers=True, data=df)
plt.legend(title='')
plt.ylabel('Train RMSE')
plt.xlabel('Number of inducing points, p')
# plt.yscale('log')
plt.title('$n=400$')
plt.tight_layout()
