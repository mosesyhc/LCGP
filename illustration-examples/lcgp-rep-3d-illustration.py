import numpy as np
import tensorflow as tf
from lcgp import LCGP
from lcgp import evaluation
import pathlib

def make_hartmann_tf(alpha, A, P, a=0.0, b=1.0, axis=-1):
    def hartmann(x):
        r = tf.reduce_sum(A * tf.square(x - P), axis=axis)
        return (a - tf.reduce_sum(alpha * tf.exp(-r), axis=-1)) / b
    return hartmann

outputdir = r'illustration-examples/rep/'
pathlib.Path(outputdir).mkdir(exist_ok=True)

# Hartmann 6D params
alpha = np.array([1.0, 1.2, 3.0, 3.2])

A = {}
A[6] = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
                ])

P = {}
P[6] = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                        [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650],
                        [4047, 8828, 8732, 5743, 1091,  381]])

x_min = {}
x_min[6] = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])

axis = {}
axis[6] = -1

a = {}
a[6] = 0.0

b = {}
b[6] = 1.0

# Create Hartmann 6D function
hartmann6d = make_hartmann_tf(alpha, A[6], P[6], a[6], b[6], axis[6])

n = 3
ntest = 1000

xtrain_uniq = np.random.uniform(0, 1, (n, 6))
xtest = np.random.uniform(0, 1, (ntest, 6))

reps = np.array((1, 2, 3))
xtrain_replicated = np.repeat(xtrain_uniq, reps, axis=0)

U = np.zeros((sum(reps), len(reps)))
for i in range(len(reps)):
    U[sum(reps[:i]):sum(reps[:(i+1)]), i] = 1

def generate_hartmann_data(x, noisy=True, noise_level=0.01):
    x_tf = tf.constant(x, dtype=tf.float32)
    
    y_base = hartmann6d(x_tf).numpy()
    
    y1 = y_base  
    y2 = 0.5 * y_base + 2.0 * np.sum(x, axis=1) - 1.0
    y3 = -0.8 * y_base - 3.0 * np.prod(x, axis=1) + 2.0  
    
    if noisy:
        noise1 = np.random.normal(0, noise_level * 0.1, y1.shape)
        noise2 = np.random.normal(0, noise_level * 0.2, y2.shape)
        noise3 = np.random.normal(0, noise_level * 0.3, y3.shape)
        
        y1 += noise1
        y2 += noise2
        y3 += noise3

    y = np.column_stack((y1, y2, y3))
    return y.T  

ytrue = generate_hartmann_data(xtest, noisy=False)
generating_noises_var = np.array([0.005, 0.1, 0.3]) * ytrue.var(1)

ytrain = generate_hartmann_data(xtrain_replicated, noisy=True, noise_level=0.01)
ytest = generate_hartmann_data(xtest, noisy=True, noise_level=0.01)

data = {
    'xtrain': xtrain_replicated,
    'xtest': xtest,
    'ytrain': ytrain.T,  
    'ytest': ytest.T,
    'ytrue': ytrue.T,
    'noisevars': generating_noises_var
}

model = LCGP(y=ytrain.T, 
             x=xtrain_replicated,
             U=U)  

model.fit()
predmean, predvar = model.predict(xtest)

rmse = evaluation.rmse(ytrue.T, predmean)
nrmse = evaluation.normalized_rmse(ytrue.T, predmean)
pcover, pwidth = evaluation.intervalstats(ytest.T, predmean, predvar)
dss = evaluation.dss(ytrue.T, predmean, predvar, use_diag=True)

print(f"RMSE: {rmse}")
print(f"Normalized RMSE: {nrmse}")
print(f"Prediction Coverage: {pcover}")
print(f"Prediction Width: {pwidth}")
print(f"DSS: {dss}")