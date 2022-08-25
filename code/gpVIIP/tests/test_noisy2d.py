from func2d import forrester2008
import numpy as np
import torch
import matplotlib.pyplot as plt

x = np.random.uniform(0, 1, 50)
y = forrester2008(x)

xtest = np.random.uniform(0, 1, 20)
