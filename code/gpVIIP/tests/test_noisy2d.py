from func2d import forrester2008
import numpy as np
import torch
import matplotlib.pyplot as plt
from mvn_elbo_autolatent_model import MVN_elbo_autolatent


x = np.random.uniform(0, 1, 25)
f = forrester2008(x)

xtest = np.linspace(0, 1, 500)
ftest = forrester2008(xtest, noisy=False)

# plt.plot(xtest, ftest)
# plt.scatter(x, f[:, 0])
# plt.scatter(x, f[:, 1])

x = torch.tensor(x).unsqueeze(1)
f = torch.tensor(f)

