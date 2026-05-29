import numpy as np


def rmse(yhat, ytrue):
    yhat = np.asarray(yhat, dtype=np.float64)
    ytrue = np.asarray(ytrue, dtype=np.float64)
    return float(np.sqrt(np.mean((yhat - ytrue) ** 2)))


def rmse_per_output(yhat, ytrue):
    yhat = np.asarray(yhat, dtype=np.float64)
    ytrue = np.asarray(ytrue, dtype=np.float64)
    return np.sqrt(np.mean((yhat - ytrue) ** 2, axis=1))


def summarize_metrics(yhat, ytrue):
    per_output = rmse_per_output(yhat, ytrue)
    out = {
        "rmse": rmse(yhat, ytrue),
    }
    for i, val in enumerate(per_output, start=1):
        out[f"rmse_y{i}"] = float(val)
    return out