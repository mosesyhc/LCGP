import matplotlib.pyplot as plt
plt.style.use(['science', 'no-latex', 'grid'])
plt.rcParams.update({'font.size': 14,
                     'lines.markersize': 12})

gpcolor = r'#7b3294'
gpshade = r'#c2a5cf'
vicolor = r'#008837'
vishade = r'#a6dba0'

import torch
import numpy as np

from func3d import forrester2008

# Set my noise level
noise = 1

# In[3]:


n = 250
x = np.random.uniform(0, 1, n)
# x = np.linspace(0, 1, n)

np.random.seed(150)
torch.manual_seed(1)

f = forrester2008(x, noisy=True)

x = torch.tensor(x).unsqueeze(1)
f = torch.tensor(f)

from lcgp import LCGP


model = LCGP(y=f, x=x, q=3)
model.compute_aux_predictive_quantities()
model.fit(verbose=True)

yhat0 = model.predict(x)[0]

fig, ax = plt.subplots(1, 3, figsize=(12, 5)) #, sharey='row')
for j in range(model.q):
    ax[0].scatter(x, model.g.detach()[j], marker='.', label=noise, alpha=0.5)
    ax[0].set_ylabel('$g(x)$')
    ax[0].set_xlabel('$x$')
ax[0].legend(labels=['$g_1$', '$g_2$', '$g_3$'])

for j in range(model.p):
    ax[1].scatter(x, model.y_orig.detach()[j], marker='.', label=noise, alpha=0.5)
    ax[1].set_ylabel('$f(x)$')
    ax[1].set_xlabel('$x$')
ax[1].legend(labels=['$f_1$', '$f_2$', '$f_3$'])
plt.tight_layout()

for j in range(model.p):
    ax[2].scatter(x, model.y_orig.detach()[j] - yhat0.detach()[j], marker='.', label=noise, alpha=0.5)
    ax[2].set_ylabel('$f(x)$')
    ax[2].set_xlabel('$x$')
ax[2].legend(labels=['$\hat{f}_1$', '$\hat{f}_2$', '$\hat{f}_3$'])
plt.tight_layout()

