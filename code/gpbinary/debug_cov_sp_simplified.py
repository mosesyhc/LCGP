import torch
from mvn_elbo_autolatent_sp_model import MVN_elbo_autolatent_sp
from fayans_support import read_only_complete_data

torch.set_default_dtype(torch.double)

torch.autograd.set_detect_anomaly(True)

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


nepoch_nn = 100
nepoch_elbo = 300
ntrain = 50
ntest = 50
kap = 20

f, x0, theta = read_only_complete_data(r'code/data/')

f = torch.tensor(f)
x0 = torch.tensor(x0)
theta = torch.tensor(theta)

m, n = f.shape  # nloc, nparam
d = theta.shape[1]

tempind = torch.randperm(n)
tr_inds = tempind[:ntrain]
te_inds = tempind[-ntest:]

torch.manual_seed(0)
ftr = f[:, tr_inds]
thetatr = theta[tr_inds]
fte = f[:, te_inds]
thetate = theta[te_inds]

# choose inducing points
ni = 20
kmeans_theta = KMeans(n_clusters=ni).fit(thetatr)
thetai = torch.Tensor(kmeans_theta.cluster_centers_)
D = pairwise_distances(thetatr, thetai)
overlap_inds = torch.where(torch.tensor(D) == 0)[1]
print(overlap_inds)
# for i in overlap_inds:
#     thetai[i] += torch.normal(torch.zeros(d), 0.1 * thetai.std(0))

psi = ftr.mean(1).unsqueeze(1)

x = torch.column_stack((x0[:, 0], x0[:, 1],
                        *[x0[:, 2] == k for k in torch.unique(x0[:, 2])]))
F = ftr - psi
# use SVD to save time
Phi_match, _, _ = torch.linalg.svd(F, full_matrices=False)
Phi_match = Phi_match[:, :kap]

mse_Phi = torch.mean((Phi_match @ Phi_match.T @ F - F) ** 2)
print('Reproducing Phi0 error in prediction of F: ', mse_Phi)

Phi = Phi_match
print('Basis size: ', Phi.shape)
lmb = torch.Tensor(0.5 * torch.log(torch.Tensor([theta.shape[1]])) +
                   torch.log(torch.std(theta, 0)))
lmb = torch.cat((lmb, torch.Tensor([0])))
Lmb = lmb.repeat(kap, 1)
Lmb[:, -1] = torch.log(torch.var(Phi.T @ (F - psi), 1))
lsigma2 = torch.Tensor(torch.log(mse_Phi))
model = MVN_elbo_autolatent_sp(Lmb=Lmb, initLmb=True,
                 lsigma2=lsigma2, psi=torch.zeros_like(psi),
                 Phi=Phi, F=F, theta=thetatr, thetai=thetai)
model.double()

###########################################################
### compare full covariance vs sparse covariance results
from matern_covmat import covmat
from mvn_elbo_autolatent_sp_model import cov_sp

# C = covmat(theta[:m], theta[:m], Lmb[0])
# C_inv = torch.linalg.inv(C)
# logdet_C = torch.linalg.slogdet(C).logabsdet
# C_sp, C_sp_inv, logdet_C_sp = cov_sp(theta[:m], theta[100:125], Lmb[0])
#
# print(C.shape)
# print(((C - C_sp)**2).sum())
# print(((C_inv - C_sp_inv)**2).sum())
# print(logdet_C, logdet_C_sp)
###########################################################

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=10e-3)
header = ['iter', 'neg elbo', 'test mse', 'train mse']
print('\nELBO training:')
print('{:<5s} {:<12s} {:<12s} {:<12s}'.format(*header))
for epoch in range(nepoch_elbo):
    optim.zero_grad()
    negelbo = model.negelbo()
    negelbo.backward(retain_graph=True)
    optim.step()  # lambda: model.lik())

    mse = model.test_mse(thetate, fte - psi)
    trainmse = model.test_mse(thetatr, F)

    print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f}'.format
          (epoch, negelbo, mse, trainmse))


