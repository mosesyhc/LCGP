import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys, io, contextlib, time, json, tracemalloc
import numpy as np
import tensorflow as tf
import lcgp.lcgp as L
from lcgp import LCGP

NOISE_STD = np.array([0.05, 0.08, 0.10])
lam_fn = lambda x: np.exp(1.5 * np.sin(2 * np.pi * np.asarray(x)))


def f_true(x):
    x = np.asarray(x, dtype=np.float64)
    return np.vstack([0.8 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * x,
                      0.3 + 0.5 * np.cos(2 * np.pi * x),
                      -0.4 - (x - 0.5) ** 2 + 0.2 * np.sin(4 * np.pi * x)])


def make(n, m, seed):
    rng = np.random.default_rng(seed)
    xu = np.linspace(0, 1, n)
    x = np.repeat(xu, m)
    sd = NOISE_STD[:, None] * np.sqrt(lam_fn(x))[None]
    y = f_true(x) + rng.normal(size=sd.shape) * sd
    return x[:, None], y


# count objective evaluations inside fit
_calls = []
_orig_min = L.scipy_minimize
def _counting_min(closure, variables, **kw):
    res = _orig_min(closure, variables, **kw)
    _calls.append(int(res.nfev))
    return res
L.scipy_minimize = _counting_min


def eval_time(m, reps=5):
    vs = m.trainable_variables
    def once():
        with tf.GradientTape() as tape:
            l = m.loss()
        tape.gradient(l, vs)
    once()  # trace
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); once(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def run(n, m, seed=0, submethod='rep', hetero=True, do_fit=True):
    x, y = make(n, m, seed)
    xt = np.linspace(0, 1, 300)[:, None]
    with contextlib.redirect_stdout(io.StringIO()):
        t = time.perf_counter()
        mdl = LCGP(y=y, x=x, q=3, submethod=submethod, hetero_noise=hetero,
                   diag_error_structure=[1, 1, 1], robust_mean=True)
        t_init = time.perf_counter() - t
        t_eval = eval_time(mdl)
        rec = dict(n=n, m=m, N=n * m, submethod=submethod, hetero=hetero,
                   t_init=t_init, t_eval=t_eval)
        if not do_fit:
            return rec
        _calls.clear()
        t = time.perf_counter(); mdl.fit(); t_fit = time.perf_counter() - t
        t = time.perf_counter(); mu, pv, cv = [a.numpy() for a in mdl.predict(xt)]
        t_pred = time.perf_counter() - t
    ft = f_true(xt[:, 0]); sd = NOISE_STD[:, None] * np.sqrt(lam_fn(xt[:, 0]))[None]
    rng = np.random.default_rng(seed + 1)
    ynew = ft[None] + rng.normal(size=(20,) + ft.shape) * sd[None]
    z = 1.959964
    cover_pi = np.mean(np.abs(ynew - mu[None]) <= z * np.sqrt(pv)[None])
    cover_ci = np.mean(np.abs(ft - mu) <= z * np.sqrt(cv))
    rec.update(t_fit=t_fit, t_pred=t_pred, nfev=sum(_calls), nfev_stages=list(_calls),
               rmse_f=float(np.sqrt(np.mean((mu - ft) ** 2))),
               noise_sd_logerr=float(np.median(np.abs(np.log(np.sqrt(pv - cv) / sd)))),
               cover_pi=float(cover_pi), cover_ci=float(cover_ci))
    return rec


def show(r):
    s = (f"n={r['n']:5d} m={r['m']:3d} N={r['N']:6d} {r['submethod']:4s} hetero={r['hetero']!s:5s} "
         f"init {r['t_init']:7.3f}s  eval {1e3*r['t_eval']:8.1f}ms")
    if 't_fit' in r:
        s += (f"  fit {r['t_fit']:7.1f}s ({r['nfev']} evals)  pred {r['t_pred']:.2f}s  "
              f"RMSE {r['rmse_f']:.4f}  noiseSD|logerr| {r['noise_sd_logerr']:.3f}  "
              f"PI cov {r['cover_pi']:.3f}  CI cov {r['cover_ci']:.3f}")
    print(s, flush=True)


if __name__ == '__main__':
    which = sys.argv[1]
    out = []
    if which == 'A':      # n fixed, N grows
        for m in [2, 4, 8, 16, 32, 64]:
            out.append(run(50, m)); show(out[-1])
        for m in [2, 4, 8, 16]:   # full-data LCGP on the same data (no hetero option)
            out.append(run(50, m, submethod='full', hetero=False)); show(out[-1])
    elif which == 'B':    # m fixed, n grows
        for n in [25, 50, 100, 200, 400, 800]:
            out.append(run(n, 4)); show(out[-1])
    elif which == 'B2':   # per-eval cost only, larger n
        for n in [25, 50, 100, 200, 400, 800, 1600]:
            out.append(run(n, 4, do_fit=False)); show(out[-1])
            out.append(run(n, 4, hetero=False, do_fit=False)); show(out[-1])
    elif which == 'C':    # constructor cost vs N at fixed n
        for m in [20, 200, 2000, 20000]:
            x, y = make(50, m, 0)
            tracemalloc.start()
            with contextlib.redirect_stdout(io.StringIO()):
                t = time.perf_counter()
                LCGP(y=y, x=x, q=3, submethod='rep', hetero_noise=True)
                dt = time.perf_counter() - t
            print(f"n=50 N={50*m:6d} constructor {dt:6.2f}s", flush=True)
            out.append(dict(n=50, m=m, N=50 * m, t_init=dt))
    json.dump(out, open(f'scaling_{which}.json', 'w'), indent=1)