import numpy as np



ns = [25, 50, 100, 500, 1000]
methods = ['surmise', 'MVGP', 'MVIP']
ipfracs = [0.25, 0.125]
noiseconsts = [0.5]

combs = []
for i in range(5):
    for n in ns:
        seed = np.random.randint(1, 50000, size=1)[0]
        for method in methods:
            for noise in noiseconsts:
                if method == 'MVIP':
                    for ipfrac in ipfracs:
                        comb = np.array((n, method, seed, noise, ipfrac))
                        combs.append(comb)
                else:
                    comb = np.array((n, method, seed, noise, 1))
                    combs.append(comb)

combs = np.array(combs)
np.savetxt(r'./params/params05.txt', X=combs, fmt='%s', delimiter='\t')
