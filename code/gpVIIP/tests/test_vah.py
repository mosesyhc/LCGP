import numpy as np
import pandas as pd
import torch
from mvn_elbo_autolatent_model import MVN_elbo_autolatent


from load_file import simulation

# Use information in the map_sim_to_design file to find the corresponding design files for each simulation batch
maps_sim_design = {}
with open('../simulation_data/map_sim_to_design') as f:
    for l in f:
        l = l.split('\n')[0]
        maps_sim_design[l.split(' ')[1]]=l.split(' ')[2]

# We use a class object to load data from each simulation batch.
# sim_d*_e&_train/test_b@: here * stands for number of design points and & for number of events per design
# @ for the batch number
name = 'mean_for_300_sliced_200_events_design'

sim_d300_e200_train = simulation(f'../simulation_data/{name}',
                                 '../simulation_data/'+'sd'+name.split('mean')[-1],
                                 f'../design_data/{maps_sim_design[name]}',
                                 f'../simulation_data/nevents_design/{name}_neve.txt')

x_orig = sim_d300_e200_train.design
f_orig = sim_d300_e200_train.obs

# Load experimental data
experiment = pd.read_csv(filepath_or_buffer="../HIC_experimental_data/PbPb2760_experiment", index_col=0)
# print(experiment.keys())
# Gather what type of experimental data do we have.
exp_label = []
for i in experiment.columns:
    words = i.split('[')
    exp_label.append(words[0] + '_[' + words[1])

# retain only available experimental observables
f_drop = f_orig[exp_label]

# remove 'fluct' as observables
fluct_colnames = [col for col in f_drop.columns if 'fluct' in col]
f_drop = f_drop[f_drop.columns.drop(fluct_colnames)]

x = torch.tensor(x_orig.values)
f = torch.tensor(f_drop.values.T)

# remove any failures
f0 = f[:, ~f.isnan().any(axis=0)]
x0 = x[~f.isnan().any(axis=0)]

model = MVN_elbo_autolatent(F=f0, x=x0, pcthreshold=0.7,
                            clamping=True)

model.fit(verbose=True, max_ls=10, c1=1e-2, c2=0.7)


