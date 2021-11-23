import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt


def negloglik(beta, theta, y, N, reg=True):
    # theta is n x d
    # beta is n x (d+1)

    n = theta.shape[0]
    theta1 = np.hstack((np.ones(n).reshape((n, 1)), theta))
    z = theta1 @ beta

    nll = np.sum(np.multiply(y, np.logaddexp(np.zeros(n), -z)) + np.multiply(N - y, np.logaddexp(np.zeros(n), z)))

    return nll


def negloglikgrad(beta, theta, y, N):

    n = theta.shape[0]
    theta1 = np.hstack((np.ones(n).reshape((n, 1)), theta))
    z = theta1 @ beta

    grad = - theta1.T @ (y - N / (1 + np.exp(-z)))

    return grad


if __name__ == '__main__':
    f = np.loadtxt(r'code/f.txt')
    theta = np.loadtxt(r'code/theta.txt')

    N, n = f.shape
    d = theta.shape[1]
    y = np.isnan(f).sum(0)

    store = []
    for b in np.arange(10):
        inds = np.random.permutation(np.arange(n))
        ytrain = y[inds[:500]]
        ytest = y[inds[500:]]

        thetatrain = theta[inds[:500]]
        thetatest = theta[inds[500:]]

        # MLE of coefficients
        beta0 = np.zeros(d+1)
        op = spo.minimize(negloglik, beta0,
                          jac=negloglikgrad,
                          method='L-BFGS-B',
                          args=(thetatrain, ytrain, N))
        beta = op.x

        # in-sample prediction
        theta1train = np.hstack((np.ones(500).reshape((500, 1)), thetatrain))
        ptrain = 1 / (1 + np.exp(-theta1train @ beta))

        # out-of-sample data set
        theta1test = np.hstack((np.ones(500).reshape((500, 1)), thetatest))
        ptest = 1 / (1 + np.exp(-theta1test @ beta))

        res = {'beta': beta,
               'N': N,
               'thetatrain': thetatrain,
               'ytrain': ytrain,
               'ptrain': ptrain,
               'thetatest': thetatest,
               'ytest': ytest,
               'ptest': ptest}
        store.append(res)

    trainrmse = np.array([(np.sqrt((res['N'] * res['ptrain'] - res['ytrain']) ** 2)).mean() for res in store])
    testrmse = np.array([(np.sqrt((res['N'] * res['ptest'] - res['ytest']) ** 2)).mean() for res in store])

    # plot one of the results
    ptrain = store[0]['ptrain']
    ytrain = store[0]['ytrain']
    ptest = store[0]['ptest']
    ytest = store[0]['ytest']
    N = store[0]['N']

    plt.figure()
    plt.plot(N * ptrain, ytrain, 'x')
    plt.title('In-sample predictions')
    plt.xlabel('Predictive mean')
    plt.ylabel('Actual value')
    plt.text(125, 25, 'RMSE = {:.3f}'.format((np.sqrt((N * ptrain - ytrain) ** 2)).mean()))
    plt.show()

    plt.figure()
    plt.plot(N * ptest, ytest, 'x')
    plt.title('Out-of-sample predictions')
    plt.xlabel('Predictive mean')
    plt.ylabel('Actual value')
    plt.text(125, 25, 'RMSE = {:.3f}'.format((np.sqrt((N * ptest - ytest) ** 2)).mean()))
    plt.show()
