import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['science'])

res_file = r'C:\Users\moses\Desktop\git\binary-hd-emulator\code\test_results\testresults_mvlatent_202202180234.npy'

d = np.load(res_file, allow_pickle=True)[()]
df_Phi0 = pd.DataFrame(np.concatenate(d['optimTFFT_Phi'], axis=0),
                  columns=('run', 'seed', 'epoch', 'mse'))
df_Phi0['mode'] = r'$G$ not optimized'
df_Phi1 = pd.DataFrame(np.concatenate(d['optimTTFT_Phi'], axis=0),
                  columns=('run', 'seed', 'epoch', 'mse'))
df_Phi1['mode'] = r'$G$ optimized'
df_Phi = pd.concat((df_Phi0, df_Phi1), axis=0, ignore_index=True)

plt.figure()
sns.lineplot(x='epoch', y='mse', data=df_Phi)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel(r'$\lVert (\Phi \Phi^T - I)F \rVert^2$')
plt.ylim(10e-2, 40)
plt.tight_layout()
plt.savefig(r'code/fig/error_Phi.png', dpi=150)
plt.close()

### GP
surmise_testmse = d['surmise'][:, 2].mean()

df_G0 = pd.DataFrame(np.concatenate(d['optimTFFT_GP'], axis=0),
                     columns=('run', 'seed', 'epoch', 'loglikelihood', 'testmse', 'trainmse'))
df_G0['mode'] = r'$G$ not optimized'
df_G1 = pd.DataFrame(np.concatenate(d['optimTTFT_GP'], axis=0),
                     columns=('run', 'seed', 'epoch', 'loglikelihood', 'testmse', 'trainmse'))
df_G1['mode'] = r'$G$ optimized'
df_G = pd.concat((df_G0, df_G1), axis=0, ignore_index=True)

plt.figure()
sns.lineplot(x='epoch', y='testmse', data=df_G, hue='mode', style='mode')
plt.hlines(surmise_testmse, 0, 400, linestyles=':')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Test MSE')
plt.ylim(10e-1, 25)
plt.legend(title='')
plt.tight_layout()
plt.savefig(r'code/fig/error_GP.png', dpi=150)
plt.close()


plt.figure()
sns.lineplot(x='epoch', y='trainmse', data=df_G, hue='mode', style='mode')
plt.hlines(d['surmise'][:, 3].mean(), 0, 400, linestyles=':')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Train MSE')
plt.ylim(10e-2, 10e0)
plt.legend(title='')
plt.tight_layout()
plt.savefig(r'code/fig/error_GPtrain.png', dpi=150)
plt.close()
