import argparse
import numpy as np
import torch
from test_general import test_single
parser = argparse.ArgumentParser(description='Takes argument to mainemutest().')


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError('Input should be either True or False')


def read_test_data(dir):
    testf = np.loadtxt(dir + r'testf.txt')
    testtheta = np.loadtxt(dir + r'testtheta.txt')
    return testf, testtheta


def read_data(dir):
    f = np.loadtxt(dir + r'f.txt')
    x = np.loadtxt(dir + r'x.txt')
    theta = np.loadtxt(dir + r'theta.txt')
    return f, x, theta


parser.add_argument('--method', help='name of emulator method')
parser.add_argument('--n', type=int, help='number of parameters')
parser.add_argument('--ipfrac', type=float, help='fraction of failures')
parser.add_argument('--noiseconst', type=float, help='noise multiplier')
parser.add_argument('--seed', type=int, help='id of replication')
parser.add_argument('--fname', help='name of function, in {borehole, piston, wingweight, ')

args = parser.parse_args()

dir = r'../data/{:s}_data/'.format(args.fname)
f, x0, xtr = read_data(dir)
fte0, xte = read_test_data(dir)

m, ntr = f.shape
fstd = f.std(1)
ftr = np.zeros_like(f)

fte = np.zeros_like(fte0)
_, nte = fte.shape

noiseconst = args.noiseconst
for j in range(m):
    ftr[j] = f[j] + np.random.normal(0, noiseconst * fstd[j], ntr)
    fte[j] = fte0[j] + np.random.normal(0, noiseconst * fstd[j], nte)

ftr = torch.tensor(ftr)
fte0 = torch.tensor(fte0)
fte = torch.tensor(fte)
xtr = torch.tensor(xtr)
xte = torch.tensor(xte)

torch.manual_seed(args.seed)
tr_ind = torch.randperm(ntr)[:args.n]
ftr_n = ftr[:, tr_ind]
xtr_n = xtr[tr_ind]
torch.seed()

save_csv = True

test_single(method=args.method, fname=args.fname, n=args.n, seed=args.seed,
            ftr=ftr_n, xtr=xtr_n,
            fte=fte, fte0=fte0, xte=xte,
            noiseconst=args.noiseconst,
            rep=args.seed, ip_frac=args.ipfrac,
            output_csv=save_csv, dir='./save/res_out/')
