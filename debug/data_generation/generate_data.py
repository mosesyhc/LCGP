import numpy as np
from pathlib import Path
import pandas as pd

np.random.seed(42)

def f_true(x):
    x = np.asarray(x, dtype=np.float64)
    f1 = 0.8 + 0.3*np.sin(2*np.pi*x) + 0.2*x
    f2 = 0.3 + 0.5*np.cos(2*np.pi*x)
    f3 = -0.4 - (x-0.5)**2 + 0.2*np.sin(4*np.pi*x)
    return np.vstack([f1, f2, f3])

def make_rep_data(n_unique=12, rep_choices=(1,2,3,4), noise_std=(0.05, 0.08, 0.10), seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        choice = lambda a, size=None, replace=True: rng.choice(a, size=size, replace=replace)
        normal = lambda mu, sig: rng.normal(mu, sig)
    else:
        choice = lambda a, size=None, replace=True: np.random.choice(a, size=size, replace=replace)
        normal = lambda mu, sig: np.random.normal(mu, sig)

    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)
    r = choice(rep_choices, size=n_unique, replace=True)

    xs, ys = [], []
    for i, xi in enumerate(x_unique):
        yi_true = f_true([xi])[:, 0]
        for _ in range(int(r[i])):
            eps = np.array([normal(0, noise_std[0]),
                            normal(0, noise_std[1]),
                            normal(0, noise_std[2])], dtype=np.float64)
            xs.append([xi]); ys.append(yi_true + eps)

    xtrain = np.array(xs, dtype=np.float64)
    ytrain = np.array(ys, dtype=np.float64).T
    xtest  = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]
    ytrue  = f_true(xtest[:, 0])
    return xtrain, ytrain, xtest, ytrue

def make_rep_data_skewed(n_unique=40,
                         heavy_region=(0.20, 0.45),
                         light_rep_choices=(1, 2),
                         heavy_rep_choices=(8, 12, 16, 20),
                         noise_std=(0.05, 0.08, 0.10),
                         seed=None):
    rng = np.random.default_rng(seed)
    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)

    xs, ys = [], []
    for xi in x_unique:
        r = int(rng.choice(heavy_rep_choices) if (heavy_region[0] <= xi <= heavy_region[1])
                else rng.choice(light_rep_choices))
        yi_base = f_true([xi])[:, 0]
        for _ in range(r):
            eps = np.array([rng.normal(0, noise_std[0]),
                            rng.normal(0, noise_std[1]),
                            rng.normal(0, noise_std[2])], dtype=np.float64)
            xs.append([xi]); ys.append(yi_base + eps)

    xtrain = np.array(xs, dtype=np.float64)
    ytrain = np.array(ys, dtype=np.float64).T
    xtest  = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]
    ytrue  = f_true(xtest[:, 0])
    return xtrain, ytrain, xtest, ytrue

def make_rep_data_hotspots(n_unique=50,
                           hotspots=((0.15, 10, 15), (0.50, 18, 25), (0.80, 12, 20)),
                           base_rep_choices=(1,),
                           noise_std=(0.05, 0.08, 0.10),
                           seed=None):
    rng = np.random.default_rng(seed)
    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)
    hotspot_idx = {np.argmin(np.abs(x_unique - x0)): (lo, hi) for (x0, lo, hi) in hotspots}

    xs, ys = [], []
    for i, xi in enumerate(x_unique):
        if i in hotspot_idx:
            lo, hi = hotspot_idx[i]
            r = int(rng.integers(lo, hi + 1))
        else:
            r = int(rng.choice(base_rep_choices))
        yi_base = f_true([xi])[:, 0]
        for _ in range(r):
            eps = np.array([rng.normal(0, noise_std[0]),
                            rng.normal(0, noise_std[1]),
                            rng.normal(0, noise_std[2])], dtype=np.float64)
            xs.append([xi]); ys.append(yi_base + eps)

    xtrain = np.array(xs, dtype=np.float64)
    ytrain = np.array(ys, dtype=np.float64).T
    xtest  = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]
    ytrue  = f_true(xtest[:, 0])
    return xtrain, ytrain, xtest, ytrue

# --------------------------
# Choose dataset
# --------------------------
# CASE 1: Uniform-ish replication
# results_fig_path = './results_figure_rep_1d_uniform/'
# Path(results_fig_path).mkdir(parents=True, exist_ok=True)
# xtrain, ytrain, xtest, ytrue = make_rep_data(
#     n_unique=16,
#     rep_choices=(1,2,3,4,5),
#     noise_std=(0.05, 0.08, 0.10),
#     seed=2025
# )

# # CASE 2: Skewed replication
results_fig_path = './results_figure_rep_1d_skewed/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)
xtrain, ytrain, xtest, ytrue = make_rep_data_skewed(
    n_unique=40,
    heavy_region=(0.20, 0.45),
    light_rep_choices=(1, 2),
    heavy_rep_choices=(8, 12, 16, 20),
    noise_std=(0.05, 0.08, 0.10),
    seed=123
)

# CASE 3: Hot-spots
# results_fig_path = './results_figure_rep_1d_hotspots/'
# Path(results_fig_path).mkdir(parents=True, exist_ok=True)
# xtrain, ytrain, xtest, ytrue = make_rep_data_hotspots(
#     n_unique=50,
#     hotspots=((0.15, 10, 15), (0.50, 18, 25), (0.80, 12, 20)),
#     base_rep_choices=(1,),
#     noise_std=(0.05, 0.08, 0.10),
#     seed=7
# )


# --------------------------
# Save data as CSV
# --------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Training Data
train_df = pd.DataFrame(
    np.hstack([xtrain, ytrain.T]),
    columns=["x", "y1", "y2", "y3"]
)

train_path = DATA_DIR / "train_data.csv"
train_df.to_csv(train_path, index=False)

# Testing Data
test_df = pd.DataFrame(
    np.hstack([xtest, ytrue.T]),
    columns=["x", "y1_true", "y2_true", "y3_true"]
)

test_path = DATA_DIR / "test_data.csv"
test_df.to_csv(test_path, index=False)

print(f"Training data saved to: {train_path}")
print(f"Test data saved to: {test_path}")

