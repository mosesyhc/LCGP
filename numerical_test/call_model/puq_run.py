import time
import numpy as np
from numerical_test.call_model.base_run import BaseModelRun


class PUQRun(BaseModelRun):

    def __init__(self, runno, data, covtype="Gaussian", verbose=False):
        super().__init__(runno=runno, data=data, model_name="PUQ-hetGP", verbose=verbose)
        self.covtype = covtype
        self._models = []

    def define_model(self):
        try:
            from hetgpy import hetGP  
        except ImportError as e:
            raise ImportError(
                "hetGPy not installed."
            ) from e
        self._models = []

    def train(self):
        if self._models is None:
            raise RuntimeError("define model first!")

        try:
            from hetgpy import hetGP
        except ImportError as e:
            raise ImportError("hetGPy not installed.") from e

        X = np.asarray(self.xtrain, dtype=np.float64)  
        Y = np.asarray(self.ytrain, dtype=np.float64)  
        p = Y.shape[0]

        t0 = time.perf_counter()
        self._models = []
        for k in range(p):
            gp = hetGP()
            gp.mle(
                X=X,
                Z=Y[k],         
                covtype=self.covtype,
            )
            self._models.append(gp)
        self.fit_time = time.perf_counter() - t0

    def predict(self, train=False, return_fullcov=False):
        if not self._models:
            raise RuntimeError("train first!")

        if return_fullcov:
            raise NotImplementedError("bad covariance")

        x0 = self.xtrain if train else self.xtest
        x0 = np.asarray(x0, dtype=np.float64)  
        p  = len(self._models)
        n0 = x0.shape[0]

        ymean    = np.zeros((p, n0), dtype=np.float64)
        ypredvar = np.zeros((p, n0), dtype=np.float64)
        yconfvar = np.zeros((p, n0), dtype=np.float64)

        t0 = time.perf_counter()
        for k, gp in enumerate(self._models):
            preds = gp.predict(x=x0)
            ymean[k]    = preds["mean"].ravel()
            yconfvar[k] = preds["sd2"].ravel()                        
            ypredvar[k] = (preds["sd2"] + preds["nugs"]).ravel()     
        self.pred_time = time.perf_counter() - t0

        result = (ymean, ypredvar, yconfvar)
        self.last_prediction = result
        return result