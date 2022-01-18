import numpy as np


def borehole(theta):
    """
    Wraps the borehole function
    """
    rw = theta[0]
    r = theta[1]
    Tu = theta[2]
    Hu = theta[3]
    Tl = theta[4]
    Hl = theta[5]
    L = theta[6]
    Kw = theta[7]
    frac1 = 2 * np.pi * Tu * (Hu - Hl)
    frac2a = (2 * L * Tu) / (np.log(r / rw) * (rw ** 2) * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1 + frac2a + frac2b)
    f = frac1 / frac2
    return f

def gen_true_theta():
    """Generate one parameter to be the true parameter for calibration."""
    thetalimits = np.array([[0.05, 0.15], #rw
                            [100, 50000], # r
                            [63070, 115600], # Tu
                            [990, 1110], # Hu
                            [63.1, 116], # Tl
                            [700, 820], # Hl
                            [1120, 1680], # L
                            [9855, 12045]]) # Kw
    theta =  np.random.uniform(thetalimits[:,0],
                               thetalimits[:,1])
    return theta

# Generate Design
theta = np.zeros((100,8))
f = np.zeros((100,1))
for k in range(0,100):
    theta[k,:]= gen_true_theta()
    f[k] = borehole(theta[k,:])
mv = np.mean(f)
f = f - mv

thetatest = np.zeros((50, 8))
ftest = np.zeros((50, 1))
for k in range(0, 50):
    thetatest[k, :] = gen_true_theta()
    ftest[k] = borehole(thetatest[k, :])-mv


################ TORCH ###################
import torch as tch

thetatch = tch.from_numpy(theta)
natscl = tch.reciprocal(tch.max(thetatch,0).values-
                                     tch.min(thetatch,0).values)
thetatch= thetatch * natscl

ytch = tch.from_numpy(f)
natscy = tch.sqrt(tch.var(ytch))
ytch = ytch / natscy

thetatesttch = tch.from_numpy(thetatest)  * natscl
ytesttch = tch.from_numpy(ftest) / natscy

n = thetatch.size(0)
d = thetatch.size(1)
D = tch.DoubleTensor(n,n,d)

sca = tch.nn.Sequential(
            tch.nn.Linear(8, 8),
            tch.nn.Tanh(),
            tch.nn.Linear(8, 8),
            tch.nn.Tanh(),
            tch.nn.Linear(8, 8),
            tch.nn.Tanh())

sca.double()
#sca[0].weight = tch.nn.Parameter(
#    tch.diag(0.1*tch.reciprocal(tch.max(thetatch,0).values-
#                                     tch.min(thetatch,0).values)))
#sca[2].weight = tch.nn.Parameter(
#    tch.diag(tch.reciprocal(tch.max(thetatch,0).values-
#                                     tch.min(thetatch,0).values)))
#sca[4].weight = tch.nn.Parameter(
 #   tch.diag(tch.reciprocal(tch.max(thetatch,0).values-
 #                                    tch.min(thetatch,0).values)))
#sca = tch.nn.Linear(8, 8)
#sca.double()
#sca.weight = tch.nn.Parameter(
#    tch.diag(0.1*tch.reciprocal(tch.max(thetatch,0).values-
#                                     tch.min(thetatch,0).values)))

thetasc = sca(thetatch)

def diffc(theta_,D_):
    for k in range(0,theta_.size(1)):
        D_[:,:,k] = tch.abs(theta_[:,k].reshape(-1, 1) -
                           theta_[:,k])


R = tch.DoubleTensor(n,n)
def matern(D,R):
    R[:] = tch.sum(-D+tch.log(1+D),2)
    R[:] = tch.exp(R[:])

def matern2(D,R):
    R[:] = tch.sum(-D+tch.log(1+D),2)
    R[:] = tch.exp(R[:])

diffc(thetasc,D)
matern(D,R)

n2 = thetatesttch.size(0)
Dtest = tch.DoubleTensor(n2,n,d)
def diffc2(theta1_,theta2_, D_):
    for k in range(0, theta1_.size(1)):
        D_[:, :, k] = tch.abs(theta1_[:, k].reshape(-1, 1) -
                              theta2_[:, k])
thetatestsc = sca(thetatesttch)
diffc2(thetatestsc,thetasc,Dtest)
Rtest = tch.DoubleTensor(n2,n)
matern2(Dtest,Rtest)

def lik(theta,sca,y):
    n = theta.size(0)
    d = theta.size(1)
    D = tch.DoubleTensor(n,n,d)
    R = tch.DoubleTensor(n, n)
    diffc(sca(thetatch), D)
    matern(D, R)
    term = R.size(0)*tch.log(y.T @ tch.linalg.solve(R,y))
    term = term + tch.linalg.slogdet(R)[1]
    return term
#
L = lik(thetatch,sca,ytch)
tch.autograd.set_detect_anomaly(True)
L.backward(retain_graph=True)

optimizer = tch.optim.Adam(sca.parameters(), lr=10 ** (-2))
for t in range(100):
    optimizer.zero_grad()
    L = lik(thetatch,sca,ytch)
    L.backward(retain_graph=True)
    optimizer.step()
    if t % 5 == 0:
        print(t, L.item())
        thetatestsc = sca(thetatesttch)
        thetasch = sca(thetatch)
        diffc(sca(thetatch), D)
        matern(D, R)
        diffc2(thetatestsc,thetasch,Dtest)
        matern2(Dtest,Rtest)
        yhat = tch.matmul(Rtest, tch.linalg.solve(R, ytch))
        print(tch.sum(tch.square(ytesttch-yhat)))
#
#     loss = lik(thetatch,thetasc,D,R,sca,ytch)
#     if t % 100 == 99:
#         print(t, loss.item())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#for epoch in range(n_epochs):
    #L = lik(thetatch, thetasc, D, R, sca, ytch)
