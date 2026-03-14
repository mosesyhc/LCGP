import time
import numpy as np
from numerical_test.call_model.base_run import BaseModelRun


class PUQRun(BaseModelRun):

    def __init__(self, runno, data, verbose=False):
        super().__init__(runno=runno, data=data, model_name="PUQ-multigetgp", verbose=verbose)

    def define_model(self):
        try:
            pass
        except ImportError as e:
            raise ImportError(
                "PUQ / multigetgp is not installed or import path is different."
            ) from e

        self.model = {
            "name": "puq_placeholder_model",
        }

    def train(self):
        if self.model is None:
            raise RuntimeError("Call define_model() first.")

        t0 = time.perf_counter()

        # EDIT THIS
        self._train_mean = np.mean(self.ytrain, axis=1, keepdims=True)

        self.fit_time = time.perf_counter() - t0

    def predict(self, train=False, return_fullcov=False):
        if self.model is None:
            raise RuntimeError("Call define_model() first.")

        x0 = self.xtrain if train else self.xtest

        t0 = time.perf_counter()

        # EDIT THIS 
        ymean = np.repeat(self._train_mean, repeats=x0.shape[0], axis=1)
        ypredvar = np.zeros_like(ymean)
        yconfvar = np.zeros_like(ymean)

        self.pred_time = time.perf_counter() - t0

        result = (ymean, ypredvar, yconfvar)
        self.last_prediction = result
        return result