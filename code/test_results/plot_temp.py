import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt(r'C:\Users\moses\Desktop\git\binary-hd-emulator\code\test_results\temp_elbo_ipGP.txt', skiprows=1)

fig, ax = plt.subplots(1, 2)
ax[0].plot(d[:, 0], d[:, 1], label='negative elbo')
ax[0].set_xlabel('Training Epoch')
ax[0].set_ylabel('Negative ELBO')
ax[1].plot(d[:, 0], d[:, 2], 'b:', label='test error')
ax[1].plot(d[:, 0], d[:, 3], 'r--', label='train error')
ax[1].set_xlabel('Training Epoch')
ax[1].set_ylabel('Error')
ax[1].legend()
plt.tight_layout()
plt.show()