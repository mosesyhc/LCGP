import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['science'])

res_file = r'C:\Users\moses\Desktop\git\binary-hd-emulator\code\test_results\elbo_20220308\testresults_mvnelbo_202203091744.npy'
dtext = r'202203091744'
d = np.load(res_file, allow_pickle=True)[()]
df_Phi0 = pd.DataFrame(np.concatenate(d['optim_Phi'], axis=0),
                  columns=('run', 'seed', 'epoch', 'mse'))
df_Phi0['mode'] = r'$\mu-v-$'
df_Phi1 = pd.DataFrame(np.concatenate(d['optim_Phi_autolatent'], axis=0),
                  columns=('run', 'seed', 'epoch', 'mse'))
df_Phi1['mode'] = r'$\mu+v-$'
df_Phi = pd.concat((df_Phi0, df_Phi1), axis=0, ignore_index=True)

plt.figure()
sns.lineplot(x='epoch', y='mse', data=df_Phi)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel(r'$\lVert (\Phi \Phi^T - I)F \rVert^2$')
plt.ylim(1, 40)
plt.tight_layout()
plt.savefig(r'code/fig/error_Phi_{:s}.png'.format(dtext), dpi=150)
plt.close()

### GP
surmise_testmse = d['surmise'][:, 2].mean()
surmise_trainmse = d['surmise'][:, 3].mean()

df_G0 = pd.DataFrame(np.concatenate(d['optim_elbo'], axis=0),
                     columns=('run', 'seed', 'epoch', 'negelbo', 'testmse', 'trainmse'))
df_G0['mode'] = r'$\mu = \Phi^T F, V = 0$'
df_G1 = pd.DataFrame(np.concatenate(d['optim_elbo_autolatent'], axis=0),
                     columns=('run', 'seed', 'epoch', 'negelbo', 'testmse', 'trainmse'))
df_G1['mode'] = r'$\hat{\mu}, \hat{V}$'
# df_G1outlier = df_G1.iloc[:250, :]
# df_G1outlier['mode'] = r'$\hat{\mu}, \hat{V}$ outlier'
# df_G1 = df_G1.iloc[250:, :]
df_G = pd.concat((df_G0, df_G1), axis=0, ignore_index=True)  # , df_G1outlier

plt.figure()
sns.lineplot(x='epoch', y='testmse', data=df_G, hue='mode', style='mode')
plt.hlines(surmise_testmse, 0, max(df_G['epoch']), linestyles=':')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Test MSE')
plt.ylim(0, 40)
plt.legend(title='')
plt.tight_layout()
plt.savefig(r'code/fig/error_elbo_{:s}.png'.format(dtext), dpi=150)
plt.close()


plt.figure()
sns.lineplot(x='epoch', y='trainmse', data=df_G, hue='mode', style='mode')
plt.hlines(surmise_trainmse, 0, max(df_G['epoch']), linestyles=':')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Train MSE')
plt.ylim(0.5, 20)
plt.legend(title='')
plt.tight_layout()
plt.savefig(r'code/fig/error_elbotrain_{:s}.png'.format(dtext), dpi=150)
plt.close()
