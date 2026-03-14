import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# TRUE UNDERLYING  FUNCTION
# =============================================================================
def f_true(x):
    x = np.asarray(x, dtype=np.float64)
    f1 = 0.8 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * x
    f2 = 0.3 + 0.5 * np.cos(2 * np.pi * x)
    f3 = -0.4 - (x - 0.5) ** 2 + 0.2 * np.sin(4 * np.pi * x)
    return np.vstack([f1, f2, f3])


# =============================================================================
# NOISE DESIGN
# =============================================================================
def get_noise_std_same_across_x(xi, base_noise_std=(0.05, 0.08, 0.10)):
    """
    CASE 1:
        SAME NOISE ACROSS x
    """
    return np.array(base_noise_std, dtype=np.float64)


def get_noise_std_different_across_x(xi, base_noise_std=(0.05, 0.08, 0.10)):
    """
    CASE 2:
        DIFFERENT NOISE ACROSS x
    """
    x = float(xi)

    mult = (
        0.8
        + 0.6 * np.exp(-((x - 0.35) / 0.12) ** 2)
        + 1.2 * np.exp(-((x - 0.75) / 0.08) ** 2)
    )

    return np.array(base_noise_std, dtype=np.float64) * mult


# =============================================================================
# REPLICATION DESIGN
# =============================================================================
def get_reps_no_rep(x_unique):
    """
    REPLICATION A:
        NO REPLICATION
    """
    return np.ones(len(x_unique), dtype=np.int32)


def get_reps_equal_rep(x_unique, rep_count=4):
    """
    REPLICATION B:
        EQUAL REPLICATION PER UNIQUE x
    """
    return np.full(len(x_unique), int(rep_count), dtype=np.int32)


def get_reps_unequal_rep_designed(x_unique):
    """
    REPLICATION C:
        UNEQUAL / DESIGNED REPLICATION PER UNIQUE x
    """
    reps = np.ones(len(x_unique), dtype=np.int32)

    for i, xi in enumerate(x_unique):
        x = float(xi)

        # moderate replication in [0.18, 0.35]
        if 0.18 <= x <= 0.35:
            reps[i] = 4

        # heavy replication in [0.48, 0.58]
        elif 0.48 <= x <= 0.58:
            reps[i] = 12

        # smaller hotspot near x ~ 0.80
        elif abs(x - 0.80) <= 0.03:
            reps[i] = 8

        # elsewhere: single observation
        else:
            reps[i] = 1

    return reps

def expand_dataset_from_design(
    x_unique,
    reps,
    noise_std_fn,
    base_noise_std=(0.05, 0.08, 0.10),
    seed=2026,
):
    rng = np.random.default_rng(seed)

    xs = []
    ys = []

    for i, xi in enumerate(x_unique):
        yi_true = f_true([xi])[:, 0]

        local_noise_std = noise_std_fn(xi, base_noise_std=base_noise_std)

        for _ in range(int(reps[i])):
            eps = np.array(
                [
                    rng.normal(0.0, local_noise_std[0]),
                    rng.normal(0.0, local_noise_std[1]),
                    rng.normal(0.0, local_noise_std[2]),
                ],
                dtype=np.float64,
            )
            xs.append([xi])
            ys.append(yi_true + eps)

    xtrain = np.array(xs, dtype=np.float64)       
    ytrain = np.array(ys, dtype=np.float64).T     

    xtest = np.linspace(0.0, 1.0, 400, dtype=np.float64)[:, None]
    ytrue = f_true(xtest[:, 0])

    return xtrain, ytrain, xtest, ytrue, x_unique[:, None], reps

def make_dataset(
    noise_mode,
    replication_mode,
    n_unique=50,
    equal_rep_count=4,
    base_noise_std=(0.05, 0.08, 0.10),
    seed=2026,
):
    x_unique = np.linspace(0.0, 1.0, n_unique, dtype=np.float64)

    # -------------------------------------------------------------------------
    # SELECT NOISE STRUCTURE
    #
    # noise_mode == "same"
    #   -> SAME noise across x
    #
    # noise_mode == "different"
    #   -> DIFFERENT noise across x
    # -------------------------------------------------------------------------
    if noise_mode == "same":
        noise_std_fn = get_noise_std_same_across_x
    elif noise_mode == "different":
        noise_std_fn = get_noise_std_different_across_x
    else:
        raise ValueError("noise_mode must be 'same' or 'different'.")

    # -------------------------------------------------------------------------
    # SELECT REPLICATION STRUCTURE
    #
    # replication_mode == "no_rep"
    #   -> no replication
    #
    # replication_mode == "equal_rep"
    #   -> equal replication at every unique x
    #
    # replication_mode == "unequal_rep"
    #   -> designed unequal replication over x
    # -------------------------------------------------------------------------
    if replication_mode == "no_rep":
        reps = get_reps_no_rep(x_unique)
    elif replication_mode == "equal_rep":
        reps = get_reps_equal_rep(x_unique, rep_count=equal_rep_count)
    elif replication_mode == "unequal_rep":
        reps = get_reps_unequal_rep_designed(x_unique)
    else:
        raise ValueError("replication_mode must be 'no_rep', 'equal_rep', or 'unequal_rep'.")

    return expand_dataset_from_design(
        x_unique=x_unique,
        reps=reps,
        noise_std_fn=noise_std_fn,
        base_noise_std=base_noise_std,
        seed=seed,
    )

def save_dataset(out_dir, xtrain, ytrain, xtest, ytrue, x_unique, reps):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        np.hstack([xtrain, ytrain.T]),
        columns=["x", "y1", "y2", "y3"],
    )

    test_df = pd.DataFrame(
        np.hstack([xtest, ytrue.T]),
        columns=["x", "y1_true", "y2_true", "y3_true"],
    )

    rep_df = pd.DataFrame(
        np.hstack([x_unique, reps.reshape(-1, 1)]),
        columns=["x_unique", "rep_count"],
    )

    train_df.to_csv(out_dir / "train_data.csv", index=False)
    test_df.to_csv(out_dir / "test_data.csv", index=False)
    rep_df.to_csv(out_dir / "replication_structure.csv", index=False)

def generate_all_datasets(base_dir="numerical_test/results/data", seed=2026):
    """
    Generate exactly these 6 dataset types:

    1) same noise across x, no replication
    2) same noise across x, equal replication per unique x
    3) same noise across x, unequal replication per unique x

    4) different noise across x, no replication
    5) different noise across x, equal replication per unique x
    6) different noise across x, unequal replication per unique x
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    dataset_specs = {
        # ---------------------------------------------------------------------
        # 1) SAME NOISE ACROSS x + NO REPLICATION
        # ---------------------------------------------------------------------
        "same_noise__no_rep": {
            "noise_mode": "same",
            "replication_mode": "no_rep",
        },

        # ---------------------------------------------------------------------
        # 2) SAME NOISE ACROSS x + EQUAL REPLICATION
        # ---------------------------------------------------------------------
        "same_noise__equal_rep": {
            "noise_mode": "same",
            "replication_mode": "equal_rep",
        },

        # ---------------------------------------------------------------------
        # 3) SAME NOISE ACROSS x + UNEQUAL REPLICATION
        # ---------------------------------------------------------------------
        "same_noise__unequal_rep": {
            "noise_mode": "same",
            "replication_mode": "unequal_rep",
        },

        # ---------------------------------------------------------------------
        # 4) DIFFERENT NOISE ACROSS x + NO REPLICATION
        # ---------------------------------------------------------------------
        "different_noise__no_rep": {
            "noise_mode": "different",
            "replication_mode": "no_rep",
        },

        # ---------------------------------------------------------------------
        # 5) DIFFERENT NOISE ACROSS x + EQUAL REPLICATION
        # ---------------------------------------------------------------------
        "different_noise__equal_rep": {
            "noise_mode": "different",
            "replication_mode": "equal_rep",
        },

        # ---------------------------------------------------------------------
        # 6) DIFFERENT NOISE ACROSS x + UNEQUAL REPLICATION
        # ---------------------------------------------------------------------
        "different_noise__unequal_rep": {
            "noise_mode": "different",
            "replication_mode": "unequal_rep",
        },
    }

    for dataset_name, spec in dataset_specs.items():
        xtrain, ytrain, xtest, ytrue, x_unique, reps = make_dataset(
            noise_mode=spec["noise_mode"],
            replication_mode=spec["replication_mode"],
            n_unique=50,
            equal_rep_count=4,
            base_noise_std=(0.05, 0.08, 0.10),
            seed=seed,
        )

        save_dataset(
            out_dir=base_dir / dataset_name,
            xtrain=xtrain,
            ytrain=ytrain,
            xtest=xtest,
            ytrue=ytrue,
            x_unique=x_unique,
            reps=reps,
        )

        print(f"Saved dataset: {dataset_name} -> {base_dir / dataset_name}")


if __name__ == "__main__":
    generate_all_datasets()