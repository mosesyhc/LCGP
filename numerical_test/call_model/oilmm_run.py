import time
import numpy as np
from numerical_test.call_model.base_run import BaseModelRun


class OILMMRun(BaseModelRun):
    def __init__(self, runno, data, num_latent=3, verbose=False):
        super().__init__(runno=runno, data=data, model_name="OILMM", verbose=verbose)
        self.num_latent = num_latent
        self.posterior = None

    def define_model(self):
        try:
            import tensorflow as tf
            from stheno import GP, EQ
            from oilmm.tensorflow import OILMM
        except ImportError as e:
            raise ImportError(
                "OILMM import failed. Make sure oilmm, stheno, probmods, and plum-dispatch "
                "are installed in a compatible environment."
            ) from e

        # Current benchmark code looks 1D (xtest[:, 0], etc). Keep this explicit.
        if self.xtrain.shape[1] != 1:
            raise NotImplementedError(
                f"OILMMRun currently supports only 1D inputs, got d={self.xtrain.shape[1]}"
            )

        self._tf = tf
        self._OILMM = OILMM

        def build_latent_processes(ps):
            return [
                (
                    p.variance.positive(1.0) * GP(EQ().stretch(p.length_scale.positive(1.0))),
                    p.noise.positive(1e-2),
                )
                for p, _ in zip(ps, range(self.num_latent))
            ]

        self.model = OILMM(
            tf.float64,
            build_latent_processes,
            num_outputs=self.ytrain.shape[0],
        )

    def train(self):
        if self.model is None:
            raise RuntimeError("Call define_model() first.")

        x = np.asarray(self.xtrain[:, 0], dtype=np.float64)
        y = np.asarray(self.ytrain.T, dtype=np.float64)  # (n, p)

        t0 = time.perf_counter()
        self.model.fit(x, y, trace=self.verbose, jit=False)
        self.posterior = self.model.condition(x, y)
        self.fit_time = time.perf_counter() - t0

    def predict(self, train=False, return_fullcov=False):
        if self.posterior is None:
            raise RuntimeError("Call train() first.")

        x0 = self.xtrain if train else self.xtest
        x0 = np.asarray(x0[:, 0], dtype=np.float64)

        t0 = time.perf_counter()
        mean, var = self.posterior.predict(x0)
        self.pred_time = time.perf_counter() - t0

        ymean = np.asarray(mean, dtype=np.float64).T
        ypredvar = np.asarray(var, dtype=np.float64).T
        yconfvar = ypredvar.copy()

        result = (ymean, ypredvar, yconfvar)
        self.last_prediction = result
        return result