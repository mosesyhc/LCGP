import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science', 'grid'])

time_ip_400 = np.loadtxt('time_ip_400.txt')
time_ip_100 = np.loadtxt('time_ip_100.txt')
time_ip_50 = np.loadtxt('time_ip_50.txt')
time_ip_200 = np.loadtxt('time_ip.txt')
time_full = np.loadtxt('time_full.txt')

for o in [time_full, time_ip_200, time_ip_50, time_ip_100, time_ip_400]:
    o -= o[0]

t = np.arange(time_full.shape[0])

plt.figure(figsize=(8, 6))
plt.scatter(t, time_full, label=r'Full, $n=400$', marker='x')
plt.scatter(t, time_ip_400, label=r'IP, $p=n$', marker='.')
plt.scatter(t, time_ip_200, label=r'IP, $p=n/2$', marker='+')
plt.scatter(t, time_ip_100, label=r'IP, $p=n/4$', marker='^')
plt.scatter(t, time_ip_50, label=r'IP, $p=n/8$', marker='D')


plt.yscale('log')
plt.xlabel('ELBO iteration')
plt.ylabel('time (s)')
plt.legend()
plt.tight_layout()
plt.savefig('time_IP_plot.png', dpi=150)
