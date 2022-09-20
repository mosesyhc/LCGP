import numpy as np


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
    def __init__(self, X, y):
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
            model[i] = {}
            if np.unique(y[i]).size < 1.5:
                model[i]['monoclass'] = True
                model[i]['model'] = np.unique(y[i])[0]
            else:
                modeli = LogisticRegression()
                modeli.fit(X, y[i])
                model[i]['monoclass'] = False
                model[i]['model'] = modeli
        self.model = model
        return

    def predict(self, X):
        model = self.model
        m = self.m

        npred = X.shape[0]
        ypred = np.zeros((m, npred))
        for i in np.arange(m):
            if model[i]['monoclass']:
                ypred[i] = model[i]['model'] * np.ones(npred)
            else:
                ypred[i] = model[i]['model'].predictmean(X)
        return ypred


if __name__ == '__main__':
    f0, x0, theta0 = read_data(r'../data/')
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
    print(r'Missing percentages: {:.3f} (Training), {:.3f} (Testing)'.format(ytr.mean(), yte.mean()))
    # plot data
    # visualize_dataset(ytr, yte)

    lrmodel = mvLogisticRegression(thetatr, ytr)
    ypred = lrmodel.predict(thetatr)

    print(r'RMSE: {:.3f}, Accuracy: {:.3f}'.format(((ypred - yte)**2).mean(), (ypred == yte).mean()))