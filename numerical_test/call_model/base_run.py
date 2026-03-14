import time
import numpy as np
from numerical_test.metrics import summarize_metrics


class BaseModelRun:
    def __init__(self, runno, data, model_name, verbose=False):
        self.runno = runno
        self.data = data
        self.model_name = model_name
        self.verbose = verbose

        self.xtrain = np.asarray(data["xtrain"], dtype=np.float64)
        self.ytrain = np.asarray(data["ytrain"], dtype=np.float64)
        self.xtest = np.asarray(data["xtest"], dtype=np.float64)

        self.ytrue = data.get("ytrue", data.get("ytest", None))
        if self.ytrue is not None:
            self.ytrue = np.asarray(self.ytrue, dtype=np.float64)

        self.model = None
        self.fit_time = None
        self.pred_time = None
        self.last_prediction = None

        self._shape_check()

    def _shape_check(self):
        if self.xtrain.ndim != 2:
            raise ValueError(f"xtrain must be 2D, got {self.xtrain.shape}")
        if self.ytrain.ndim != 2:
            raise ValueError(f"ytrain must be 2D, got {self.ytrain.shape}")
        if self.ytrain.shape[1] != self.xtrain.shape[0]:
            raise ValueError(
                f"ytrain shape must be (p, N) with N == xtrain.shape[0]. "
                f"Got ytrain={self.ytrain.shape}, xtrain={self.xtrain.shape}"
            )
        if self.xtest.ndim != 2:
            raise ValueError(f"xtest must be 2D, got {self.xtest.shape}")
        if self.xtest.shape[1] != self.xtrain.shape[1]:
            raise ValueError(
                f"xtest input dimension mismatch: xtest.shape[1]={self.xtest.shape[1]} "
                f"vs xtrain.shape[1]={self.xtrain.shape[1]}"
            )

    @staticmethod
    def _to_numpy(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def define_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self, train=False, return_fullcov=False):
        raise NotImplementedError

    def evaluate(self, ymean=None):
        if self.ytrue is None:
            raise ValueError("No ytrue in provided data.")

        if ymean is None:
            if self.last_prediction is None:
                raise RuntimeError("No cached prediction. Call predict() first.")
            ymean = self.last_prediction[0]

        metric_dict = summarize_metrics(ymean, self.ytrue)
        metric_dict.update(
            {
                "runno": self.runno,
                "model": self.model_name,
                "fit_time_sec": self.fit_time,
                "pred_time_sec": self.pred_time,
                "total_time_sec": (
                    None if (self.fit_time is None or self.pred_time is None)
                    else self.fit_time + self.pred_time
                ),
            }
        )
        return metric_dict

    def run_all(self, return_fullcov=False):
        self.define_model()
        self.train()
        pred = self.predict(train=False, return_fullcov=return_fullcov)
        metrics = self.evaluate(ymean=pred[0])

        return {
            "metrics": metrics,
            "prediction": pred,
        }