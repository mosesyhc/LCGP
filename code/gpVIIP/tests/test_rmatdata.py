import torch
import numpy as np


f = np.load(r'C:\Users\cmyh\Documents\git\binary-hd-emulator\code\data\rmat_data\f.npy')
theta = np.load(r'C:\Users\cmyh\Documents\git\binary-hd-emulator\code\data\rmat_data\theta.npy')
x = np.load(r'C:\Users\cmyh\Documents\git\binary-hd-emulator\code\data\rmat_data\x.npy')

thetastd = (theta - theta.min(0)) / (theta.max(0) - theta.min(0))

