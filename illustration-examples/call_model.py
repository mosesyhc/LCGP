from lcgp import LCGP
import numpy as np


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
    def __init__(self, submethod='full', robust=True, err_struct=None,
                 num_latent=None, var_threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.modelname = 'LCGP'
        self.num_latent = num_latent
        self.var_threshold = var_threshold
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
                          var_threshold=self.var_threshold,
                          diag_error_structure=self.err_struct,
                          robust_mean=self.robust,
                          submethod=self.submethod)

    def train(self):
        self.model.fit(verbose=self.verbose)

    def predict(self, train: bool = False, return_fullcov: bool = False, as_pxn: bool = False):
        xtest = self.xtrain if train else self.xtest
        out = self.model.predict(xtest, return_fullcov=return_fullcov)

        if return_fullcov:
            ymean, ypredvar, yconfvar, yfullpredcov = out
            ymean = ymean.numpy()
            ypredvar = ypredvar.numpy()
            yconfvar = yconfvar.numpy()
            yfullpredcov = yfullpredcov.numpy()
            if as_pxn:  # return (p, n0) if you want to match your recomposition code
                ymean = ymean.T
                ypredvar = ypredvar.T
                yconfvar = yconfvar.T
            return ymean, ypredvar, yconfvar, yfullpredcov
        else:
            ymean, ypredvar, yconfvar = out
            ymean = ymean.numpy()
            ypredvar = ypredvar.numpy()
            yconfvar = yconfvar.numpy()
            if as_pxn:
                ymean = ymean.T
                ypredvar = ypredvar.T
                yconfvar = yconfvar.T
            return ymean, ypredvar, yconfvar


def rmse(ytrue, yhat):
    return float(np.sqrt(np.mean((ytrue - yhat)**2)))

def normalized_rmse(ytrue, yhat, method="range"):
    if method == "range":
        ranges = np.ptp(ytrue, axis=1, keepdims=True)  
        ranges = np.where(ranges == 0, 1.0, ranges)
        nrmse_per = np.sqrt(np.mean((ytrue - yhat)**2, axis=1, keepdims=True)) / ranges
        return float(np.mean(nrmse_per))
    elif method == "std":
        stds = np.std(ytrue, axis=1, ddof=0, keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        nrmse_per = np.sqrt(np.mean((ytrue - yhat)**2, axis=1, keepdims=True)) / stds
        return float(np.mean(nrmse_per))
    else:
        raise ValueError("method must be 'range' or 'std'")

def intervalstats(ytrue, mean, var, z=1.96):
    """
    95% nominal predictive interval coverage/width over all dims/points.
    Use FUNCTION variance (var = confvar) if comparing to noise-free truth.
    """
    sd = np.sqrt(var)
    lo, hi = mean - z*sd, mean + z*sd
    covered = (ytrue >= lo) & (ytrue <= hi)
    cover = float(np.mean(covered))
    width = float(np.mean(2*z*sd))
    return cover, width

def dss(ytrue, mean, var, use_diag=True):
    """
    Dawid–Sebastiani score (Gaussian): (y-μ)^2 / σ^2 + log σ^2
    Aggregated over dims/points; use function variance if scoring latent f.
    """
    eps = 1e-12
    s2 = np.maximum(var, eps)
    term = ((ytrue - mean)**2) / s2 + np.log(s2)
    return float(np.mean(term))
