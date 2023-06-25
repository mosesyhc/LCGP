import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mvn_elbo_autolatent_model import MVN_elbo_autolatent

from load_file import simulation

# VAH root
vah_root = r'..\..\git\VAH_SURMISE'
# Use information in the map_sim_to_design file to find the corresponding design files for each simulation batch
maps_sim_design = {}
with open(vah_root + '\simulation_data\map_sim_to_design') as f:
    for l in f:
        l = l.split('\n')[0]
        maps_sim_design[l.split(' ')[1]] = l.split(' ')[2]

# We use a class object to load data from each simulation batch.
# sim_d*_e&_train/test_b@: here * stands for number of design points and & for number of events per design
# @ for the batch number
name = 'mean_for_300_sliced_200_events_design'

sim_d300_e200_train = simulation(vah_root + f'/simulation_data/{name}',
                                 vah_root + '/simulation_data/'+'sd'+name.split('mean')[-1],
                                 vah_root + f'/design_data/{maps_sim_design[name]}',
                                 vah_root + f'/simulation_data/nevents_design/{name}_neve.txt')

x_orig = sim_d300_e200_train.design
f_orig = sim_d300_e200_train.obs
f_sd_orig = sim_d300_e200_train.obs_sd

plt.style.use(['science', 'grid', 'high-vis'])
plt.scatter(np.arange(f_sd_orig.shape[1]), f_sd_orig.mean(0),
            marker='x')
plt.xlabel('response')
plt.ylabel('response stdev')

# Load experimental data
experiment = pd.read_csv(filepath_or_buffer=vah_root + r"\HIC_experimental_data\PbPb2760_experiment", index_col=0)
# print(experiment.keys())
# Gather what type of experimental data do we have.
exp_label = []
for i in experiment.columns:
    words = i.split('[')
    exp_label.append(words[0] + '_[' + words[1])

# retain only available experimental observables
f_avail = f_orig[exp_label]
f_sd_avail = f_sd[exp_label]

# remove 'fluct' as observables
fluct_colnames = [col for col in f_avail.columns if 'fluct' in col]
f_avail = f_avail[f_avail.columns.drop(fluct_colnames)]
f_sd_avail = f_sd_avail[f_sd_avail.columns.drop(fluct_colnames)]

x = torch.tensor(x_orig.values)
f = torch.tensor(f_avail.values.T)

trind = torch.randperm(x.shape[0])[:290].numpy()
teind = np.setdiff1d(np.arange(x.shape[0]), trind)
xtr = x[trind]
xte = x[teind]
ftr = f[:, trind]
fte = f[:, teind]
#
# from surmise.emulation import emulator
# emu = emulator(x=np.arange(ftr.shape[0]), theta=xtr.numpy(), f=ftr.numpy(),
#                method='PCGPwM', args={'epsilon': 20})
#
# emupred = emu.predict(x=np.arange(ftr.shape[0]), theta=xte.numpy())
# emumean = emupred.mean()
# emurmse = np.sqrt(((fte.numpy() - emumean)**2).mean())

model = MVN_elbo_autolatent(F=ftr, x=xtr, kap=25,
                            clamping=True)

model.fit(verbose=True)
predtr = model(xtr)
predte = model(xte)

r2 = 1 - ((ftr - predtr)**2).sum(1) / (((ftr.T - ftr.mean(1)).T)**2).sum(1)
r2te = 1 - ((fte - predte) **2).sum(1) / (((fte.T - fte.mean(1)).T)**2).sum(1)
# r2emutr = 1 - ((ftr - emu.predict().mean())**2).sum(1) / (((ftr.T - ftr.mean(1)).T)**2).sum(1)
# r2emute = 1 - ((fte - emumean)**2).sum(1) / (((fte.T - fte.mean(1)).T)**2).sum(1)
# plt.scatter(np.arange(98), r2emutr.detach().numpy())
