import numpy as np

fnames = ['borehole', 'piston', 'wingweight', 'otlcircuit']
ns = [100, 250, 500]
noiseconsts = [0.25]
methods = ['MVGP', 'surmise']
ipfracs = [1]

base = np.stack(np.meshgrid(ns, methods, noiseconsts, ipfracs, fnames)).reshape(5, -1).T

combs = []
rep = 5
for i in range(rep):
    nseed = len(ipfracs) * len(noiseconsts) * len(ns) * len(fnames)
    seeds = np.random.randint(0, 100000, nseed)

    comb = np.insert(base, 2, np.tile(seeds, len(methods)), axis=1)
    combs.extend(comb)

np.savetxt('params_gen2.txt', combs, delimiter='\t', fmt='%s')