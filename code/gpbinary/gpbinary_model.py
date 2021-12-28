import numpy as np
import scipy.stats as sps
import torch


def read_data(dir):
    f = np.loadtxt(dir + r'f.txt')
    x = np.loadtxt(dir + r'x.txt')
    theta = np.loadtxt(dir + r'theta.txt')
    return f, x, theta


def visualize_dataset(ytrain, ytest):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(-ytrain.T, aspect='auto', cmap='gray', interpolation='none')
    ax[1].imshow(-ytest.T, aspect='auto', cmap='gray', interpolation='none')
    ax[0].set_title('Training data')
    ax[0].set_ylabel('Parameters')
    ax[1].set_title('Testing data')
    plt.show()


class mvLogisticRegression():
    def __init__(self, y, X):
        self.X = X
        self.y = y
        self.m, self.n = y.shape
        self.model = None
        self.fit()

    def fit(self):
        from sklearn.linear_model import LogisticRegression
        X = self.X
        y = self.y
        m = self.m
        model = {}
        for i in np.arange(m):
            modeli = LogisticRegression()
            modeli.fit(X, y[i])
            model[i] = modeli
        self.model = model
        return

    def predict(self, X):
        model = self.model
        m = self.m

        ypred = np.zeros((m, X.shape[0]))
        for i in np.arange(m):
            ypred[i] = model[i].predict(X)
        return ypred


def ind_logistic_reg(y, theta):
    from sklearn.linear_model import LogisticRegression

    model = {}
    for i in np.arange(y.shape[0]):
        m = LogisticRegression()
        m.fit(theta, y[i])
        model[i] = m
    return model

def get_psi(y):
    z = (y.sum(1) + 10) / (y.shape[1] + 20)
    psi = sps.norm.ppf(z)
    return psi


def get_Phi(x):
    tmp = x[:, :2]
    tmp[:, 0] -= tmp[:, 1]  # Use (N, Z) instead of (A, Z)
    Phi = (tmp - tmp.mean(0)) / tmp.std(0)
    return Phi


if __name__ == '__main__':
    f0, x0, theta0 = read_data(r'code/data/')
    y0 = np.isnan(f0).astype(int)

    # choose training and testing data
    failinds = np.argsort(y0.sum(0))
    traininds = failinds[-250:-50][::2]
    testinds = failinds[-250:-50][1::2]

    ytr = y0[:, traininds]
    thetatr = theta0[traininds]
    yte = y0[:, testinds]
    thetate = theta0[testinds]

    # percent missing
    print(r'Missing percentages: {:.3f} (Training), {:.3f} (Testing)'.format(ytrain.mean(), ytest.mean()))
    # plot data
    visualize_dataset(ytr, yte)

    psi = get_psi(ytr)
    Phi = get_Phi(x0)

