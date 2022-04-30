import pandas as pd
import seaborn as sns

df = pd.read_csv(r'C:\Users\moses\Desktop\git\binary-hd-emulator\code\test_results\kmeans_trial.csv')
df['lsigma2'] = df['lsigma2'].astype(str)
# df['label'] = 'n = ' + (1/df['ip_frac']).astype(int).astype(str) + 'p'
for ip_frac in df['ip_frac'].unique():
    sns.lineplot(x='n', y='d_mean_I', hue='lsigma2', data=df)
