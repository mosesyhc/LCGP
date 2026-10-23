"""
Stochastic SIR calibration example: LCGP version of David O'Gara's hetGP walkthrough.
"""

from adaptive_ts.example_simulator.odinSIR.odin import run_sim
from adaptive_ts.utils import make_grid
from scipy.stats import qmc
from lcgp import LCGP
import lcgp.lcgp as _lcgp_mod
import inspect, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['mathtext.fontset'] = 'cm'

seed_ground_truth = np.arange(1,10).reshape(-1,1) + 5_000
pars_ground_truth = np.array([[0.75,0.25]])
pars = make_grid(pars_ground_truth,seed_ground_truth)

sims = pd.concat(
    [run_sim(par).assign(sim_id=i+1) for i,par in enumerate(pars)]
)
fig, ax = plt.subplots()
sim_agg = sims.groupby('T')['n_SI'].mean().reset_index()
for sim_id, sim in sims.groupby('sim_id'):
    sim.plot(x='T',y='n_SI',alpha=0.4,color='gray',ax=ax,label=None,legend=False)
sim_agg.plot(x='T',y='n_SI',color='black',ax=ax,label=None,legend=False)
ground_truth = sim_agg.copy()
ax.set_xlabel('Time')
ax.set_ylabel('New Infections')
plt.show()

rng = np.random.default_rng(123)
lhs = qmc.LatinHypercube(d = 2, rng = rng)
seeds = np.arange(1,11).reshape(-1,1)
pars = make_grid(lhs.random(n=250),seeds)
X = pars[:,0:2]
sims = pd.concat(
    [run_sim(par).assign(sim_id=i+1) for i,par in enumerate(pars)]
)
sims_matrix = sims.pivot_table(index='sim_id',columns='T',values='n_SI').values

T_grid = np.sort(sims['T'].unique())
Xu = np.unique(X, axis=0)
print('X shape:', X.shape, '| sims_matrix shape:', sims_matrix.shape,
      '| unique X rows:', Xu.shape[0])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].scatter(Xu[:, 0], Xu[:, 1], s=12, alpha=0.7, label='LHS design (x10 reps)')
axes[0].scatter(*pars_ground_truth[0], color='red', marker='*', s=200, label='Truth')
axes[0].set_xlabel(r'Transmission rate $\beta$')
axes[0].set_ylabel(r'Recovery rate $\gamma$')
axes[0].set_title('Design')
axes[0].legend(fontsize=8)

axes[1].plot(T_grid, sims_matrix.T, color='tab:blue', alpha=0.03)
axes[1].plot(ground_truth['T'], ground_truth['n_SI'], color='black', lw=2)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('New Infections')
axes[1].set_title(f'All {sims_matrix.shape[0]} training runs')

pick = np.random.default_rng(0).choice(len(Xu), size=4, replace=False)
for j, k in enumerate(pick):
    rows = np.all(X == Xu[k], axis=1)
    axes[2].plot(T_grid, sims_matrix[rows].T, color=f'C{j}', alpha=0.35)
    axes[2].plot([], [], color=f'C{j}',
                 label=f'beta={Xu[k,0]:.2f}, gamma={Xu[k,1]:.2f}')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('New Infections')
axes[2].set_title('Replicates at 4 design points')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

_src = textwrap.dedent(inspect.getsource(LCGP._compute_aux_predictive_quantities_rep))
if 'sigma_inv_sqrt_used[:, None]  # (q,p)' in _src:
    _src = _src.replace('sigma_inv_sqrt_used[:, None]  # (q,p)',
                        'sigma_inv_sqrt_used[None, :]  # (q,p)')
    _ns = {}
    exec(_src, _lcgp_mod.__dict__, _ns)
    LCGP._compute_aux_predictive_quantities_rep = _ns['_compute_aux_predictive_quantities_rep']

y_train = sims_matrix.T.astype(np.float64)        # (n_time, 2500)
x_train = X.astype(np.float64)                    # (2500, 2)

model = LCGP(y=y_train, x=x_train, var_threshold=0.99, submethod='rep')
model.fit()
print('latent components q =', model.q)

observed = ground_truth['n_SI'].values           

Xgrid = lhs.random(n=20_000)

pred_mean, pred_var = [], []
for s in range(0, len(Xgrid), 2_000):
    m, pv, _ = model.predict(Xgrid[s:s + 2_000])
    pred_mean.append(np.asarray(m).T)
    pred_var.append(np.asarray(pv).T)
pred_mean = np.vstack(pred_mean)            
pred_var = np.vstack(pred_var)

ABC = rng.normal(loc = pred_mean, scale = np.maximum(pred_var, 0)**0.5, size = pred_mean.shape)

dist = ((ABC - observed)**2).mean(axis=1)**0.5
threshold = np.quantile(dist, 0.01)
print(f'threshold = {threshold:.2f}, accepted = {(dist <= threshold).sum()}')

post = pd.DataFrame(Xgrid[dist <= threshold,:],columns = ['beta','gamma'])

g = sns.jointplot(post,x='beta',y='gamma',kind='kde',xlim=[0,1],ylim=[0,1],fill=True)
g.ax_joint.set_xlabel(r'Transmission rate $\beta$')
g.ax_joint.set_ylabel(r'Recovery rate $\gamma$')
g.ax_joint.scatter(pars_ground_truth[:,0],pars_ground_truth[:,1],color='red',marker='*')
plt.show()

pars_ABC = post.sample(n = 100, random_state=123)

pars_ABC['seed'] = np.arange(1,len(pars_ABC)+1) + len(pars)

sims_abc = pd.concat(
    [run_sim(par).assign(sim_id=i+1) for i,par in enumerate(pars_ABC.values)]
)
sims_prior = sims[sims['sim_id'].isin(rng.choice(sims['sim_id'].unique(),replace=False,size=100))]
g = sns.relplot(
    pd.concat([sims_prior.assign(label='LHS'),sims_abc.assign(label='ABC')]),x='T',y='n_SI',hue='label',
    units = 'sim_id', alpha = 0.25,
    col = 'label', kind = 'line',
    estimator=None,
)
for ax in g.axes.flat:
    ax.plot(ground_truth['T'],ground_truth['n_SI'],color='black')
    ax.set_xlabel('Time')
    ax.set_ylabel('Incident Infections')
plt.show()