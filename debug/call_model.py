import numpy as np
from lcgp.lcgp_rep import LCGP


class LCGPRun:

    def __init__(self,
                 runno,
                 data,
                 num_latent=None,
                 var_threshold=None,
                 submethod="full",
                 diag_error_structure=None,
                 robust_mean=True,
                 parameter_clamp_flag=False,
                 rep_standardize_ybar=True,
                 verbose=False):

        self.runno = runno
        self.data = data

        self.xtrain = data["xtrain"]
        self.ytrain = data["ytrain"]
        self.xtest  = data["xtest"]
        self.ytest  = data["ytest"]

        self.ytrue = data.get("ytrue", None)
        self.ystd  = data.get("ystd", None)

        self.num_latent = num_latent
        self.var_threshold = var_threshold
        self.submethod = submethod
        self.diag_error_structure = diag_error_structure
        self.robust_mean = robust_mean
        self.parameter_clamp_flag = parameter_clamp_flag
        self.rep_standardize_ybar = rep_standardize_ybar
        self.verbose = verbose

        self.model = None

        self._shape_check()

    def _shape_check(self):
        x = np.asarray(self.xtrain)
        y = np.asarray(self.ytrain)

        if y.shape[1] != x.shape[0]:
            raise ValueError(
                f"ytrain shape {y.shape} must be (p,N) with N=xtrain rows {x.shape[0]}"
            )

    def define_model(self):

        if (self.num_latent is not None) and (self.var_threshold is not None):
            raise ValueError("Provide only num_latent OR var_threshold.")

        self.model = LCGP(
            y=self.ytrain,
            x=self.xtrain,
            q=self.num_latent,
            var_threshold=self.var_threshold,
            diag_error_structure=self.diag_error_structure,
            parameter_clamp_flag=self.parameter_clamp_flag,
            robust_mean=self.robust_mean,
            submethod=self.submethod,
            rep_standardize_ybar=self.rep_standardize_ybar,
            verbose=self.verbose,
        )

    def train(self):
        if self.model is None:
            raise RuntimeError("Call define_model() first.")
        self.model.fit(verbose=self.verbose)

    def predict(self, train=False, return_fullcov=False):

        if self.model is None:
            raise RuntimeError("Call define_model() first.")

        x0 = self.xtrain if train else self.xtest
        out = self.model.predict(x0=x0, return_fullcov=return_fullcov)

        if return_fullcov:
            ymean, ypredvar, yconfvar, yfullpredcov = out
            return (ymean.numpy(),
                    ypredvar.numpy(),
                    yconfvar.numpy(),
                    None if yfullpredcov is None else yfullpredcov.numpy())

        else:
            ymean, ypredvar, yconfvar = out
            return ymean.numpy(), ypredvar.numpy(), yconfvar.numpy()
