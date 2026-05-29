import time
import numpy as np
from lcgp.lcgp import LCGP
from numerical_test.call_model.base_run import BaseModelRun


class LCGPRun(BaseModelRun):
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
        model_name = f"LCGP-{submethod}"
        super().__init__(runno=runno, data=data, model_name=model_name, verbose=verbose)

        self.num_latent = num_latent
        self.var_threshold = var_threshold
        self.submethod = submethod
        self.diag_error_structure = diag_error_structure
        self.robust_mean = robust_mean
        self.parameter_clamp_flag = parameter_clamp_flag
        self.rep_standardize_ybar = rep_standardize_ybar

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

        t0 = time.perf_counter()
        self.model.fit(verbose=self.verbose)
        self.fit_time = time.perf_counter() - t0

    def predict(self, train=False, return_fullcov=False):
        if self.model is None:
            raise RuntimeError("Call define_model() first.")

        x0 = self.xtrain if train else self.xtest

        t0 = time.perf_counter()
        out = self.model.predict(x0=x0, return_fullcov=return_fullcov)
        self.pred_time = time.perf_counter() - t0

        if return_fullcov:
            ymean, ypredvar, yconfvar, yfullpredcov = out
            result = (
                self._to_numpy(ymean),
                self._to_numpy(ypredvar),
                self._to_numpy(yconfvar),
                None if yfullpredcov is None else self._to_numpy(yfullpredcov),
            )
        else:
            ymean, ypredvar, yconfvar = out
            result = (
                self._to_numpy(ymean),
                self._to_numpy(ypredvar),
                self._to_numpy(yconfvar),
            )

        self.last_prediction = result
        return result