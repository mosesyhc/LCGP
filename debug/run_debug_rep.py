import numpy as np

from lcgp import LCGP
from debug.data_generation.read_debug_data import load_train_test_csv

def main():
    data = load_train_test_csv(
        train_csv="debug/data_generation/data/train_data.csv",
        test_csv="debug/data_generation/data/test_data.csv",
    )

    xtrain = data["xtrain"]          
    ytrain = data["ytrain"]         

    xtest  = data["xtest"]
    ytrue  = data.get("ytrue", None)      

    m = LCGP(
        y=ytrain,
        x=xtrain,
        q=None,                    
        var_threshold=None,      
        submethod="rep",
        diag_error_structure=[1,1,1],
        robust_mean=True,
        rep_standardize_ybar=False,          # dont standarsize
        verbose=True,
    )

    print("\n==== REPLICATION STRUCTURES ====")
    print("x_unique:", m.x_unique.numpy().shape)
    print(m.x_unique.numpy())
    print("r:", m.r.numpy().shape, m.r.numpy())
    print("R:", m.R.numpy().shape)
    print("ybar (RAW):", m.ybar.numpy().shape)
    print(m.ybar.numpy())

    print("\n==== BASIS / LATENT ====")
    print("q:", m.q)
    print("phi:", m.phi.numpy().shape)
    print(m.phi.numpy())
    print("diag_D:", m.diag_D.numpy().shape)
    print(m.diag_D.numpy())
    print("g:", m.g.numpy().shape)
    print(m.g.numpy())

    print("\n==== HYPERPARAMETERS (raw tensors) ====")
    lLmb, lLmb0, lsigma2s, lnugGPs = m.get_param()
    print("lLmb:", m.lLmb.numpy())
    print("lLmb0:", m.lLmb0.numpy())
    print("lsigma2s(built):", lsigma2s.numpy())
    print("lnugGPs:", m.lnugGPs.numpy())

    print("\n==== FIT ====")
    m.fit(verbose=True)

    print("\n==== PREDICT ====")
    ymean, ypredvar, yconfvar = m.predict(x0=xtest, return_fullcov=False)
    print("ymean:", ymean.numpy().shape)
    print(ymean.numpy())
    print("ypredvar:", ypredvar.numpy().shape)
    print(ypredvar.numpy())
    print("yconfvar:", yconfvar.numpy().shape)
    print(yconfvar.numpy())

    if ytrue is not None:
        print("\n==== COMPARE TO ytrue ====")
        print("ytrue:", ytrue.shape)
        print(ytrue)

if __name__ == "__main__":
    main()
