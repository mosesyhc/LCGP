import pytest
import numpy as np
import tensorflow as tf
from lcgp import LCGP


# ===========================================================================
# Fixtures / shared helpers
# ===========================================================================

def _make_rep_data(seed=0, n_unique=20, p=4, d=2, reps=3):
    """
    Build a replicated dataset: each unique x row appears `reps` times.
    Returns x (N, d), y (p, N) where N = n_unique * reps.
    """
    rng = np.random.default_rng(seed)
    x_unique = rng.uniform(0, 1, (n_unique, d))
    x = np.tile(x_unique, (reps, 1))          # (N, d)
    y = rng.standard_normal((p, n_unique * reps))
    return x, y, x_unique, n_unique


def _make_full_data(seed=0, n=50, p=4, d=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal((p, n))
    return x, y


# ===========================================================================
# 1. Standardization tests
# ===========================================================================

class TestStandardizeX:
    @pytest.mark.parametrize('n,d', [(30, 1), (50, 2), (100, 3)])
    def test_range_is_zero_to_one(self, n, d):
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, (n, d))
        xs, x_min, x_max, _, _ = LCGP.init_standard_x(
            tf.convert_to_tensor(x, dtype=tf.float64)
        )
        assert float(tf.reduce_min(xs)) >= 0.0 - 1e-9
        assert float(tf.reduce_max(xs)) <= 1.0 + 1e-9

    @pytest.mark.parametrize('n,d', [(30, 1), (50, 2)])
    def test_shape_preserved(self, n, d):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, (n, d))
        xs, _, _, _, _ = LCGP.init_standard_x(
            tf.convert_to_tensor(x, dtype=tf.float64)
        )
        assert xs.shape == (n, d)

    @pytest.mark.parametrize('n,d', [(30, 2)])
    def test_xnorm_positive(self, n, d):
        """xnorm (mean pairwise distance) should be positive for non-constant inputs."""
        rng = np.random.default_rng(7)
        x = rng.uniform(0, 1, (n, d))
        _, _, _, _, xnorm = LCGP.init_standard_x(
            tf.convert_to_tensor(x, dtype=tf.float64)
        )
        assert float(tf.reduce_min(xnorm)) > 0.0


class TestStandardizeY:
    @pytest.mark.parametrize('robust_mean', [True, False])
    def test_standardized_scale(self, robust_mean):
        """After standardization, spread of ys should be ~1."""
        x, y = _make_full_data()
        model = LCGP(y=y, x=x, robust_mean=robust_mean)
        ys, _, _, _ = model.init_standard_y(
            tf.convert_to_tensor(y, dtype=tf.float64)
        )
        # each row should have spread roughly 1
        row_std = tf.math.reduce_std(ys, axis=1).numpy()
        assert np.all(row_std > 0.1), "Standardized rows have near-zero spread"

    @pytest.mark.parametrize('robust_mean', [True, False])
    def test_shape_preserved(self, robust_mean):
        x, y = _make_full_data(n=40, p=3)
        model = LCGP(y=y, x=x, robust_mean=robust_mean)
        ys, ycenter, yspread, _ = model.init_standard_y(
            tf.convert_to_tensor(y, dtype=tf.float64)
        )
        assert ys.shape == y.shape
        assert ycenter.shape == (y.shape[0], 1)
        assert yspread.shape == (y.shape[0], 1)

    @pytest.mark.parametrize('robust_mean', [True, False])
    def test_invertible(self, robust_mean):
        """tx_y(init_standard_y(y)) should recover y."""
        x, y = _make_full_data()
        model = LCGP(y=y, x=x, robust_mean=robust_mean)
        # model.ymean / model.ystd are set during __init__ on model.y (already std),
        # so test inversion directly
        yt = tf.convert_to_tensor(y, dtype=tf.float64)
        ys, center, spread, _ = model.init_standard_y(yt)
        y_recovered = ys * spread + center
        np.testing.assert_allclose(y_recovered.numpy(), y, atol=1e-10)


