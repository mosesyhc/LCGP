from lcgp import LCGP

# try:
#     import gpytorch
#     import torch
#     from torch import tensor
#     from oilmm.tensorflow import OILMM
#     from stheno import GP, Matern32
#     from surmise.emulation import emulator
# except:
#     pass


class SuperRun:
    def __init__(self, runno: str, data, verbose=False, **kwargs):
        self.data = data
        self.xtrain = data['xtrain']
        self.ytrain = data['ytrain']
        self.xtest = data['xtest']
        self.ytest = data['ytest']
        if 'ytrue' in data.keys():
            self.ytrue = data['ytrue']
        if 'ystd' in data.keys():
            self.ystd = data['ystd']
        self.runno = runno
        self.model = None
        self.modelname = ''
        self.n = self.xtrain.shape[0]
        self.num_output = self.ytrain.shape[0]
        self.verbose = verbose

        return

    def define_model(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


class LCGPRun(SuperRun):
    def __init__(self, submethod='full', robust=True, err_struct=None, **kwargs):
        super().__init__(**kwargs)
        self.modelname = 'LCGP'
        self.num_latent = kwargs['num_latent']
        self.submethod = submethod
        self.robust = robust
        self.err_struct = err_struct
        if self.robust:
            self.modelname += '_robust'

    def define_model(self):
        self.model = LCGP(y=self.ytrain,
                          x=self.xtrain,
                          parameter_clamp_flag=False,
                          q=self.num_latent,
                          diag_error_structure=self.err_struct,
                          robust_mean=self.robust,
                          submethod=self.submethod)

    def train(self):
        self.model.fit(verbose=self.verbose)

    def predict(self, train=False):
        if train:
            xtest = self.xtrain
        else:
            xtest = self.xtest
        ypredmean, ypredvar, _ = self.model.predict(xtest, return_fullcov=False)

        return ypredmean.numpy(), ypredvar.numpy()

#
# class OILMMRun(SuperRun):
#     def __init__(self, learn_transform=True, **kwargs):
#         super().__init__(**kwargs)
#         self.num_latent = kwargs['num_latent']
#         self.modelname = 'OILMM'
#         self.learn_transform = learn_transform
#         if not learn_transform:
#             self.modelname += '_no_tx'
#
#     def build_latent_processes(self, ps):
#         # Return models for latent processes, which are noise-contaminated GPs.
#         return [
#             (
#                 p.variance.positive(1) * GP(
#                     Matern32().stretch(p.length_scale.positive(1))),
#                 p.noise.positive(1e-2),
#             )
#             for p, _ in zip(ps, range(self.num_latent))
#         ]
#
#     def define_model(self):
#         self.model = OILMM(tf.float64,
#                            self.build_latent_processes,
#                            num_outputs=self.num_output,
#                            learn_transform=self.learn_transform)
#
#     def train(self):
#         prior = self.model
#         prior.fit(self.xtrain, self.ytrain.T, trace=self.verbose)
#
#     def predict(self):
#         prior = self.model
#
#         posterior = prior.condition(self.xtrain, self.ytrain.T)
#         ypredmean, ypredvar = posterior.predict(self.xtest)
#         return ypredmean.T, ypredvar.T
#
#
# class MultitaskGPModel(gpytorch.models.ApproximateGP):
#     def __init__(self, num_latent, num_output, xtrain):
#         inducing_points = xtrain
#
#         # We have to mark the CholeskyVariationalDistribution as batch
#         # so that we learn a variational distribution for each task
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
#             num_inducing_points=xtrain.shape[0], batch_shape=torch.Size([num_latent])
#         )
#
#         # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
#         # so that the output will be a MultitaskMultivariateNormal
#         # rather than a batch output
#         variational_strategy = gpytorch.variational.LMCVariationalStrategy(
#             gpytorch.variational.VariationalStrategy(
#                 self, inducing_points, variational_distribution,
#                 learn_inducing_locations=True
#             ),
#             num_tasks=num_output,
#             num_latents=num_latent,
#             latent_dim=-1
#         )
#
#         super().__init__(variational_strategy)
#
#         # The mean and covariance modules should be marked as batch
#         # so we learn a different set of hyperparameters
#         self.mean_module = gpytorch.means.ConstantMean(
#             batch_shape=torch.Size([num_latent]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latent])),
#             batch_shape=torch.Size([num_latent])
#         )
#
#     def forward(self, x):
#         # The forward function should be written as if we were dealing with each output
#         # dimension in batch
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
#
# class SVGPRun(SuperRun):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.modelname = 'SVGP'
#         self.num_latent = kwargs['num_latent']
#         self.loss = None
#
#     def define_model(self):
#         svgp = MultitaskGPModel(num_latent=self.num_latent,
#                                 num_output=self.num_output,
#                                 xtrain=tensor(self.xtrain))
#
#         likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
#             num_tasks=self.num_output,
#             has_task_noise=True,
#             rank=0)
#
#         optimizer = torch.optim.Adam([
#             {'params': svgp.parameters()},
#             {'params': likelihood.parameters()},
#         ], lr=0.1)
#
#         mll = gpytorch.mlls.VariationalELBO(likelihood, svgp, num_data=self.n)
#
#         self.model = svgp
#         self.likelihood = likelihood
#         self.mll = mll
#         self.optimizer = optimizer
#
#     def train(self, num_epoch=1000):
#         model = self.model
#         likelihood = self.likelihood
#         mll = self.mll
#         optimizer = self.optimizer
#
#         model.train()
#         likelihood.train()
#
#         for i in range(num_epoch):
#             optimizer.zero_grad()
#             output = model(tensor(self.xtrain))
#             loss = -mll(output, tensor(self.ytrain.T))
#             loss.backward()
#             optimizer.step()
#
#     @torch.no_grad()
#     def predict(self):
#         self.model.eval()
#         self.likelihood.eval()
#
#         with gpytorch.settings.fast_pred_var():
#             predictions = self.likelihood(self.model(tensor(self.xtest)))
#             svgpmean = predictions.mean
#             svgpvar = predictions.variance
#
#         return svgpmean.numpy().T, svgpvar.numpy().T
#
#
# class GPPCARun(SuperRun):
#     def __init__(self, directory=None, **kwargs):
#         super().__init__(**kwargs)
#         self.modelname = 'GPPCA'
#         self.num_latent = kwargs['num_latent']
#         self.data_dir = pathlib.WindowsPath
#         self.directory = directory
#
#     def define_model(self):
#         data_dir = r'{:s}/data'.format(self.directory) + '/{:s}'.format(
#             self.runno) + r'/'
#         pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)
#         self.data_dir = pathlib.Path(data_dir).absolute()
#
#         for k, v in self.data.items():
#             np.savetxt(data_dir + '{:s}.txt'.format(k), v)
#
#     def train(self):
#         script = r'C:\Program Files\R\R-4.3.1\bin\Rscript.exe'
#         dim_input = 'hd_input' if self.xtrain.ndim > 1 else '1d_input'
#         which_script = r'reference_code\GPPCA\gppca_{:s}.R'.format(dim_input)
#         print(which_script)
#         subprocess.call([script,
#                          which_script,
#                          str(self.data_dir),
#                          str(self.data_dir.joinpath('xtrain.txt')),
#                          str(self.data_dir.joinpath('ytrain.txt')),
#                          str(self.data_dir.joinpath('xtest.txt')),
#                          str(self.num_latent)
#                          ])
#
#     def predict(self):
#         ypredmean = np.loadtxt(self.data_dir.joinpath('ypredmean.txt'))
#         ypredvar = np.loadtxt(self.data_dir.joinpath('ypredvar.txt'))
#         return ypredmean, ypredvar
#
#
# class PCSKRun(SuperRun):
#     def __init__(self, sim_std=None, **kwargs):
#         super().__init__(**kwargs)
#         self.modelname = 'PCSK'
#         self.num_latent = kwargs['num_latent']
#         self.sim_std = sim_std
#
#     def define_model(self):
#         pass
#
#     def train(self):
#         p = self.ytrain.shape[0]
#         locations = np.arange(p).reshape((p, 1))
#         self.locations = locations
#         pcsk = emulator(x=locations, theta=self.xtrain,
#                         f=self.ytrain, method='PCSK',
#                         args={'numpcs': self.num_latent,
#                               'simsd': self.ystd}
#                         )
#         self.model = pcsk
#
#     def predict(self):
#         locations = self.locations
#         pcsk = self.model
#         predclass = pcsk.predict(x=locations, theta=self.xtest)
#         pcskmean = predclass.mean()
#         pcskvar = predclass.var()
#
#         return pcskmean, pcskvar
