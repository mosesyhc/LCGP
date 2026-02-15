import numpy as np

def load_train_test_csv(train_csv: str, test_csv: str):
    train = np.genfromtxt(train_csv, delimiter=",", names=True)
    test  = np.genfromtxt(test_csv,  delimiter=",", names=True)

    # train: columns x,y1,y2,y3
    xtrain = train["x"].reshape(-1, 1)               # (N,1)
    ytrain = np.vstack([train["y1"], train["y2"], train["y3"]])   # (p,N)

    # test: columns x,y1_true,y2_true,y3_true
    xtest = test["x"].reshape(-1, 1)
    ytrue = np.vstack([test["y1_true"], test["y2_true"], test["y3_true"]])  # (p,Ntest)

    return {
        "xtrain": xtrain,
        "ytrain": ytrain,
        "xtest":  xtest,
        "ytrue":  ytrue,
    }
