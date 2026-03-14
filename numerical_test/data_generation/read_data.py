import numpy as np
from pathlib import Path


def load_train_test_csv(train_csv: str, test_csv: str):
    train = np.genfromtxt(train_csv, delimiter=",", names=True)
    test = np.genfromtxt(test_csv, delimiter=",", names=True)

    xtrain = train["x"].reshape(-1, 1)
    ytrain = np.vstack([train["y1"], train["y2"], train["y3"]])

    xtest = test["x"].reshape(-1, 1)
    ytrue = np.vstack([test["y1_true"], test["y2_true"], test["y3_true"]])

    return {
        "xtrain": xtrain,
        "ytrain": ytrain,
        "xtest": xtest,
        "ytrue": ytrue,
    }


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    return load_train_test_csv(
        train_csv=str(dataset_dir / "train_data.csv"),
        test_csv=str(dataset_dir / "test_data.csv"),
    )