from surmise.emulation import emulator
import pickle
import numpy as np

x = np.random.random((50, 5))
theta = np.random.random((100, 8))
f = np.random.random((50, 100))

emu = emulator(x=x, theta=theta, f=f,
               method='PCGPwM')


with open('emu.surmise', 'wb') as f:
    pickle.dump(emu, f, protocol=pickle.HIGHEST_PROTOCOL)
