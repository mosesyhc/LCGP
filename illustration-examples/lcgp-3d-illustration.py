import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import gpytorch
from tqdm import tqdm
from functions import forrester2008
import tensorflow as tf
import lab as B
from matrix import Diagonal
from oilmm.tensorflow import OILMM
from stheno import EQ, GP, Matern32

from lcgp import LCGP, evaluation

plt.style.use(['science', 'no-latex', 'grid'])
plt.rcParams.update({'font.size': 14,
                     'lines.markersize': 12})

noise = 1

n = 250
x = np.linspace(0, 1, n)
xpred = x[1:] - 1/(2*n)

# np.random.seed(0)

ytrain = forrester2008(x, noisy=True, noises=[0.005, 0.1, 0.3])
truey = forrester2008(xpred, noisy=False)
newy = forrester2008(xpred, noisy=True, noises=[0.005, 0.1, 0.3])

# np.random.seed(None)

x = torch.tensor(x).unsqueeze(1)
xpred = torch.tensor(xpred).unsqueeze(1)
ytrain = torch.tensor(ytrain)

# LCGP


# save = {}
# for lLmb in [0, 10, 20, 40]:
#     for lLmb0 in [0, 5, 10, 20, 40]:
#         pc = {'lLmb': lLmb, 'lLmb0': lLmb0}
#         print(pc)

start0 = time.monotonic()

lcgp = LCGP(y=ytrain, x=x, q=3,
            parameter_clamp_flag=False)
# lcgp.compute_aux_predictive_quantities()
lcgp.fit(verbose=True)

end0 = time.monotonic()

yhat, ypredvar, yconfvar = lcgp.predict(xpred, return_fullcov=False)

print(
    evaluation.rmse(truey, yhat.numpy()),
    evaluation.normalized_rmse(truey, yhat.numpy()),
    evaluation.intervalstats(newy, yhat.numpy(), ypredvar.numpy())
)

start1 = time.monotonic()

lcgp_r = LCGP(y=ytrain, x=x, q=3,
              parameter_clamp_flag=False, robust_mean=True)
lcgp_r.compute_aux_predictive_quantities()
lcgp_r.fit(verbose=True)

end1 = time.monotonic()

yhat_r, ypredvar_r, yconfvar_r = lcgp_r.predict(xpred, return_fullcov=False)

print(
    evaluation.rmse(truey, yhat_r.numpy()),
    evaluation.normalized_rmse(truey, yhat_r.numpy()),
    evaluation.intervalstats(newy, yhat_r.numpy(), ypredvar_r.numpy())
)
#
# save[(lLmb, lLmb0)] = np.array((evaluation.rmse(truey, yhat.numpy()),
#                                evaluation.normalized_rmse(truey, yhat.numpy()),
#                                evaluation.intervalstats(newy, yhat.numpy(), ypredvar.numpy()),
#                                evaluation.rmse(truey, yhat_r.numpy()),
#                                evaluation.normalized_rmse(truey, yhat_r.numpy()),
#                                evaluation.intervalstats(newy, yhat_r.numpy(), ypredvar_r.numpy())
#                                ))


# print(save)
#################################################
# svgp
num_output = 3
num_latents = num_output


start2 = time.monotonic()
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.linspace(0, 1, int(n/1)).repeat(num_latents, 1).unsqueeze(-1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_output,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


svgp = MultitaskGPModel()
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_output,
                                                              has_task_noise=True,
                                                              rank=0)

svgp.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': svgp.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, svgp, num_data=n)

num_epochs=1000
epochs_iter = tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    optimizer.zero_grad()
    output = svgp(x)
    loss = -mll(output, ytrain.T)
    loss.backward()
    optimizer.step()
    # if i % 100 == 0:
    #     print('Loss: {:.6f}'.format(loss))

end2 = time.monotonic()

svgp.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(svgp(xpred))
    svgpmean = predictions.mean
    svgpvar = predictions.variance


print(
    evaluation.rmse(truey, svgpmean.numpy().T),
    evaluation.normalized_rmse(truey, svgpmean.numpy().T),
    evaluation.intervalstats(newy, svgpmean.numpy().T, svgpvar.numpy().T)
)



######################################
#
# # if False:
start3 = time.monotonic()
num_output = 3
def build_latent_processes(ps):
    # Return models for latent processes, which are noise-contaminated GPs.
    return [
        (
            p.variance.positive(1) * GP(Matern32().stretch(p.length_scale.positive(1))),
            p.noise.positive(1e-2),
        )
        for p, _ in zip(ps, range(num_output))
    ]

bruinsmaPrior = OILMM(tf.float64, build_latent_processes,
                      num_outputs=num_output, learn_transform=True) #np.array([1e-2]*num_output))
bruinsmaPrior.fit(x.numpy(), ytrain.T.numpy(), trace=True)

end3 = time.monotonic()

bruinsmaPosterior = bruinsmaPrior.condition(x.numpy(), ytrain.T.numpy())
bmean, bvar = bruinsmaPosterior.predict(xpred.numpy())

print(
    evaluation.rmse(truey, bmean.T),
    evaluation.normalized_rmse(truey, bmean.T),
    evaluation.intervalstats(newy, bmean.T, bvar.T)
)


bruinsmaPosterior.vs.print()
prior  = bruinsmaPosterior()
ps_lat = prior.ps.latent_processes.processes
d = Diagonal(B.stack(*(ps_lat[i].noise() for i in range(3))))
h = prior.model.mixing_matrix
noises = B.diag(prior.model.noise_matrix + B.mm(h, d, h, tr_c=True))
noises *= prior.ps.transform[0].scale()[0] ** 2
print("Learned observation noises:", noises)

###################################################
outputdir = 'illustration-examples/output_fig/'

fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey='row')
# for j in range(lcgp.q):
#     ax[0].scatter(x, lcgp.g.detach()[j], marker='.', label=noise, alpha=0.5)
#     ax[0].set_ylabel('$g(x)$')
#     ax[0].set_xlabel('$x$')
#     ax[0].plot(xpred, lcgp.ghat.detach()[j], label=noise, color='C{:d}'.format(j))
# ax[0].legend(labels=['$g_1$', '$g_2$', '$g_3$'])

for j in range(truey.shape[0]):
    ax[0].plot(xpred, truey[j], label=noise, color='k', linewidth=2)
    ax[0].set_ylabel('$f(x)$')
    ax[0].set_xlabel('$x$')
# ax[0].legend(labels=['$f_1$', '$f_2$', '$f_3$'])
    ax[0].scatter(x, ytrain[j], marker='.', alpha=0.2, color='C{:d}'.format(j))
    ax[0].plot(xpred, yhat_r.detach()[j], label=noise, color='C{:d}'.format(j))
    # ax[1].fill_between(xpred.squeeze(), (yhat - 2*yconfvar.sqrt()).detach()[j], (yhat + 2*yconfvar.sqrt()).detach()[j], alpha=0.5, color='C{:d}'.format(j))
    ax[0].fill_between(xpred.squeeze(), (yhat_r - 2*ypredvar_r.sqrt()).detach()[j],
                       (yhat_r + 2*ypredvar_r.sqrt()).detach()[j], alpha=0.15, color='C{:d}'.format(j))
    ax[0].set_ylabel(r'$\hat{f}(x)$')
    ax[0].set_xlabel('$x$')
    ax[0].set_title('LCGP, robust mean')

for j in range(lcgp.p):
    ax[1].plot(xpred, truey[j], label=noise, color='k', linewidth=2)
    ax[1].set_ylabel('$f(x)$')
    ax[1].set_xlabel('$x$')
# ax[1].legend(labels=['$f_1$', '$f_2$', '$f_3$'])
    ax[1].scatter(x, ytrain[j], marker='.', alpha=0.2, color='C{:d}'.format(j))
    ax[1].plot(xpred, yhat.detach()[j], label=noise, color='C{:d}'.format(j))
    # ax[1].fill_between(xpred.squeeze(), (yhat - 2*yconfvar.sqrt()).detach()[j], (yhat + 2*yconfvar.sqrt()).detach()[j], alpha=0.5, color='C{:d}'.format(j))
    ax[1].fill_between(xpred.squeeze(), (yhat - 2*ypredvar.sqrt()).detach()[j],
                       (yhat + 2*ypredvar.sqrt()).detach()[j], alpha=0.15, color='C{:d}'.format(j))
    ax[1].set_ylabel(r'$\hat{f}(x)$')
    ax[1].set_xlabel('$x$')
    ax[1].set_title('LCGP')

for j in range(truey.shape[0]):
    ax[2].plot(xpred, truey[j], label=noise, color='k', linewidth=2)
    ax[2].set_ylabel('$f(x)$')
    ax[2].set_xlabel('$x$')
# ax[2].legend(labels=['$f_1$', '$f_2$', '$f_3$'])
    ax[2].scatter(x, ytrain[j], marker='.', alpha=0.2, color='C{:d}'.format(j))
    ax[2].plot(xpred, bmean.T[j], label=noise, color='C{:d}'.format(j))
    ax[2].fill_between(xpred.squeeze(), (bmean.T - 2*np.sqrt(bvar.T))[j], (bmean.T + 2*np.sqrt(bvar.T))[j], alpha=0.15, color='C{:d}'.format(j))
    ax[2].set_ylabel(r'$\hat{f}(x)$')
    ax[2].set_xlabel('$x$')
    ax[2].set_title('OILMM')

plt.tight_layout()
plt.savefig(outputdir + 'output{:s}_{:s}.png'.format(time.strftime('%Y-%m-%d'), time.strftime('%H-%M')))
plt.close()


times = np.array([start0,
                 end0,
                 start1,
                 end1,
                 start2,
                 end2,
                 start3,
                 end3])

print(np.array2string(times[1:] - times[:-1], formatter={'float':lambda x: "%.4f" % x}))

if False:
    fig, ax = plt.subplots(3, 1, figsize=(4, 8), sharey='all')
    for j in range(lcgp_r.q):
        ax[j].plot(xpred, truey[j], label=noise, color='k', linewidth=2)
        # ax[j].legend(labels=['$f_1$', '$f_2$', '$f_3$'])
        ax[j].scatter(x, ytrain[j], marker='.', alpha=0.2, color='C{:d}'.format(j))
        ax[j].plot(xpred, yhat_r.detach()[j], label=noise, color='C{:d}'.format(j))
        # ax[j][1].fill_between(xpred.squeeze(), (yhat - 2*yconfvar.sqrt()).detach()[j], (yhat + 2*yconfvar.sqrt()).detach()[j], alpha=0.5, color='C{:d}'.format(j))
        ax[j].fill_between(xpred.squeeze(), (yhat_r - 2*ypredvar_r.sqrt()).detach()[j],
                           (yhat_r + 2*ypredvar_r.sqrt()).detach()[j], alpha=0.15, color='C{:d}'.format(j))
        ax[j].set_ylabel(r'$y_{:d}(x)$'.format(j+1))
        ax[j].set_xlabel('$x$')
    plt.tight_layout()
    plt.savefig(outputdir + 'lcgp_illustration.png', dpi=150)