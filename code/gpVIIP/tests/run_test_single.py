import argparse
from test_borehole import test_single
parser = argparse.ArgumentParser(description='Takes argument to mainemutest().')

def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError('Input should be either True or False')

parser.add_argument('--n', type=int, help='number of parameters')
# parser.add_argument('--function', help='name of test function')
# parser.add_argument('--failrandom', type=str2bool, help='True if failures are random (False if structured)')
# parser.add_argument('--failfraction', type=float, help='fraction of failures')
parser.add_argument('--method', help='name of emulator method')
parser.add_argument('--rep', type=int, help='id of replication')
parser.add_argument('--noiseconst', type=float, help='noise stdev multiplier')
parser.add_argument('--ipfrac', type=float, help='inducing point fraction')