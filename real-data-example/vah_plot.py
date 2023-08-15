import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import pathlib

save_flag = True

plt.style.use(['science', 'no-latex']) #, 'grid'])
plt.rcParams.update({'font.size': 20,
                     'lines.markersize': 15})
CNcolors = mpl.rcParams['axes.prop_cycle']

outputdir = 'real-data-example/output/'
figuredir = 'real-data-example/figures/_lat12/'
pathlib.Path(figuredir).mkdir(exist_ok=True)

# Get a list of all csv files in the directory
txt_files = glob.glob(outputdir + '*_lat12.csv')

# Initialize an empty DataFrame
dflist = []
columns = []
dtypes = [str, str, str, int,
          float, float, float,
          float, float, float]
# Read each CSV file and append them to the combined DataFrame
for file in txt_files:
    each_df = pd.read_csv(file, index_col=0).T
    dflist.append(each_df.values[1])
    columns = each_df.values[0]

dtype_dict = dict([(columns[i], dtypes[i]) for i in range(len(dtypes))])
df = pd.DataFrame(data=dflist, columns=columns)
df = df.astype(dtype_dict)

# Get unique values from 'Color' column
unique_models = df['modelname'].unique()
[unique_models[0], unique_models[1]] = [unique_models[1], unique_models[0]]
# Create a dictionary mapping unique colors to the 'tab10' colormap
color_mapping = dict(zip(unique_models,
                         itertools.cycle(['C2', 'C3', 'C4', 'C7', 'C5'])))
marker_mapping = dict(zip(unique_models, itertools.cycle([ 's', '^', 'v', 'X', 'P'])))

# color_mapping['LCGP'] = color_mapping['LCGP_robust']
# marker_mapping['LCGP'] = marker_mapping['LCGP_robust']

colors = df['modelname'].map(color_mapping)
styles = df['modelname'].map(marker_mapping)

df['logrmse'] = np.log(df['rmse'])
df['logtraintime'] = np.log(df['traintime'])
df['lognrmse'] = np.log(df['nrmse'])
df['logdss'] = np.log(df['dss'] + np.abs(df['dss'].min()) + 1)
df['colors'] = colors
df['styles'] = styles

groupby_df = df.groupby(by='modelname').median(numeric_only=True)

plot_ylist = {
              'rmse': 'RMSE',
              'logrmse': 'Log RMSE',
              'lognrmse': 'Log Normalized RMSE',
              'logdss': ' Log DS Score'}

lwidth = 1

# if False:
for y, yname in plot_ylist.items():
    plt.figure(figsize=(10, 10))
    # All results
    for model in unique_models:
        group_df = df[df['modelname'] == model]
        c = 'None' #if model != 'LCGP_robust' else group_df['colors'].values[0]
        if model != 'OILMM_no_tx':
            plt.scatter(y=y, x='logtraintime', c=c, edgecolors=group_df['colors'],
                        alpha=0.35, linewidths=lwidth,
                        marker=group_df['styles'].values[0], data=group_df, label=None)
    # Median values
    for model in unique_models:
        group_df = df[df['modelname'] == model]
        c = 'None' #if model != 'LCGP_robust' else group_df['colors'].values[0]
        if model != 'OILMM_no_tx':
            plt.scatter(y=groupby_df[y][model], x=groupby_df['logtraintime'][model],
                        c=c, edgecolors=group_df['colors'], linewidths=lwidth*5,
                        marker=group_df['styles'].values[0], label=model,
                        s=mpl.rcParams['lines.markersize']**2*2)
    plt.ylabel(yname)
    plt.xlabel('Log time (seconds)')
    plt.legend()
    plt.tight_layout()
    if save_flag:
        plt.savefig(figuredir + '{:s}.png'.format(y), dpi=300)
        plt.close()


subdf = df.loc[df['modelname'] != 'OILMM_no_tx', :]
boxplot_ylist = {'pcover': '95% Coverage',
                 'pwidth': '95% Predictive Interval Width'}

for y, yname in boxplot_ylist.items():
    plt.figure(figsize=(10, 10))
    sns.boxplot(y=y, x='modelname',
                data=subdf, palette=color_mapping,
                order=unique_models,
                width=0.5)
    if y == 'pcover':
        plt.axhline(y=0.95, color='black', linestyle='--', linewidth=2)
    plt.xlabel('')
    plt.ylabel(yname)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    if save_flag:
        plt.savefig(figuredir + '{:s}.png'.format(y), dpi=300)
        plt.close()
