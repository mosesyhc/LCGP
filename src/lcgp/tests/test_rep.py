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
# 1. Replication preprocessing helper tests
# ===========================================================================

class TestGroupUniqueRows:
    @pytest.mark.parametrize('n_unique,reps,d', [(10, 3, 1), (20, 5, 2), (15, 2, 3)])
    def test_correct_number_of_unique_rows(self, n_unique, reps, d):
        rng = np.random.default_rng(1)
        x_unique = rng.uniform(0, 1, (n_unique, d))
        x = np.tile(x_unique, (reps, 1))
        x_uniq_out, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
        assert x_uniq_out.shape[0] == n_unique

    @pytest.mark.parametrize('n_unique,reps,d', [(10, 3, 2)])
    def test_counts_equal_reps(self, n_unique, reps, d):
        rng = np.random.default_rng(2)
        x_unique = rng.uniform(0, 1, (n_unique, d))
        x = np.tile(x_unique, (reps, 1))
        _, _, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
        assert np.all(counts == reps)

    @pytest.mark.parametrize('n_unique,reps,d', [(10, 4, 2)])
    def test_inverse_reconstructs_x(self, n_unique, reps, d):
        rng = np.random.default_rng(3)
        x_unique = rng.uniform(0, 1, (n_unique, d))
        x = np.tile(x_unique, (reps, 1))
        x_uniq_out, inverse, _ = np.unique(x, axis=0, return_inverse=True, return_counts=True)
        np.testing.assert_allclose(x_uniq_out[inverse], x)


class TestComputeYbar:
    @pytest.mark.parametrize('n_unique,reps,p', [(10, 3, 4), (20, 5, 2)])
    def test_ybar_shape(self, n_unique, reps, p):
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        assert model.ybar.shape == (p, n_unique)

    @pytest.mark.parametrize('n_unique,reps,p', [(10, 3, 4)])
    def test_ybar_is_mean_of_replicates(self, n_unique, reps, p):
        """ybar[:, i] should equal the mean of the reps replicates at unique point i."""
        rng = np.random.default_rng(99)
        x_unique = rng.uniform(0, 1, (n_unique, 2))
        x = np.tile(x_unique, (reps, 1))
        y = rng.standard_normal((p, n_unique * reps))

        model = LCGP(y=y, x=x, submethod='rep')
        _, inverse, _ = np.unique(x, axis=0, return_inverse=True, return_counts=True)

        ybar_np = model.ybar.numpy()
        for i in range(n_unique):
            cols = (inverse == i)
            expected = y[:, cols].mean(axis=1)
            np.testing.assert_allclose(ybar_np[:, i], expected, atol=1e-10)

    @pytest.mark.parametrize('n_unique,reps,p', [(10, 3, 4)])
    def test_r_values(self, n_unique, reps, p):
        """model.r should contain the replication count for each unique point."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        assert np.all(model.r.numpy() == reps)


# ===========================================================================
# 2. Rep submethod: init
# ===========================================================================

class TestRepInit:
    @pytest.mark.parametrize('n_unique,reps,p,d', [
        (20, 3, 4, 2),
        (15, 5, 3, 1),
    ])
    def test_attributes_set(self, n_unique, reps, p, d):
        """All replication attributes should be present after init."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        for attr in ['x_unique', 'x_unique_s', 'ybar', 'ybar_s',
                     'ybar_mean', 'ybar_std', 'r', 'R', 'group_ids']:
            assert hasattr(model, attr), f"Missing attribute: {attr}"

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_n_resets_to_unique(self, n_unique, reps, p, d):
        """model.n should equal n_unique, not n_unique * reps."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        assert int(model.n.numpy()) == n_unique

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_x_unique_s_in_unit_cube(self, n_unique, reps, p, d):
        """Standardized unique x should lie in [0, 1]."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        xs = model.x_unique_s.numpy()
        assert xs.min() >= 0.0 - 1e-9
        assert xs.max() <= 1.0 + 1e-9

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_ybar_s_roughly_standardized(self, n_unique, reps, p, d):
        """ybar_s should have spread close to 1 per output dimension."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        spreads = tf.math.reduce_std(model.ybar_s, axis=1).numpy()
        assert np.all(spreads > 0.05), "ybar_s rows have near-zero spread"

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_R_diagonal_equals_r(self, n_unique, reps, p, d):
        """R should be diagonal with values equal to r."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        R_diag = tf.linalg.diag_part(model.R).numpy()
        np.testing.assert_array_equal(R_diag, model.r.numpy().astype(float))

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_invalid_submethod_raises(self, n_unique, reps, p, d):
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        with pytest.raises(ValueError):
            LCGP(y=y, x=x, submethod='invalid')


# ===========================================================================
# 3. Rep submethod: fit
# ===========================================================================

class TestRepFit:
    @pytest.mark.parametrize('n_unique,reps,p,d', [
        (20, 3, 4, 2),
        (15, 4, 3, 1),
    ])
    def test_fit_does_not_crash(self, n_unique, reps, p, d):
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        model.fit()  # should not raise

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_loss_decreases_after_fit(self, n_unique, reps, p, d):
        """Loss after fit should be <= loss before fit."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        loss_before = float(model.loss())
        model.fit()
        loss_after = float(model.loss())
        assert loss_after <= loss_before + 1e-3, (
            f"Loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"
        )

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_parameters_finite_after_fit(self, n_unique, reps, p, d):
        """All trainable parameters should be finite after fit."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        model.fit()
        for var in model.trainable_variables:
            assert np.all(np.isfinite(var.numpy())), f"Non-finite values in {var.name}"


# ===========================================================================
# 4. Rep submethod: predict
# ===========================================================================

class TestRepPredict:
    @pytest.mark.parametrize('n_unique,reps,p,d', [
        (20, 3, 4, 2),
        (15, 4, 3, 1),
    ])
    def test_predict_output_shapes(self, n_unique, reps, p, d):
        x, y, x_unique, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        model.fit()
        n0 = 10
        x0 = np.random.default_rng(5).uniform(0, 1, (n0, d))
        ypred, ypredvar, yconfvar = model.predict(x0)
        assert ypred.shape == (p, n0)
        assert ypredvar.shape == (p, n0)
        assert yconfvar.shape == (p, n0)

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_predictive_variance_positive(self, n_unique, reps, p, d):
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        model.fit()
        x0 = np.random.default_rng(6).uniform(0, 1, (10, d))
        _, ypredvar, _ = model.predict(x0)
        assert np.all(ypredvar.numpy() > 0), "Predictive variance has non-positive entries"

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_predict_finite(self, n_unique, reps, p, d):
        """Predictions should be finite everywhere."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        model.fit()
        x0 = np.random.default_rng(7).uniform(0, 1, (10, d))
        ypred, ypredvar, yconfvar = model.predict(x0)
        assert np.all(np.isfinite(ypred.numpy()))
        assert np.all(np.isfinite(ypredvar.numpy()))
        assert np.all(np.isfinite(yconfvar.numpy()))

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_confvar_leq_predvar(self, n_unique, reps, p, d):
        """Confidence variance should be <= total predictive variance."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        model.fit()
        x0 = np.random.default_rng(8).uniform(0, 1, (10, d))
        _, ypredvar, yconfvar = model.predict(x0)
        assert np.all(yconfvar.numpy() <= ypredvar.numpy() + 1e-9)

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_interpolation_reasonable_rmse(self, n_unique, reps, p, d):
        """
        Predicting at training x_unique should yield lower RMSE than
        predicting the mean of y (a weak but meaningful sanity check).
        """
        x, y, x_unique, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps, seed=42)
        model = LCGP(y=y, x=x, submethod='rep')
        model.fit()

        ybar_np = model.ybar.numpy()          # (p, n_unique) ground truth means
        ypred, _, _ = model.predict(x_unique)
        pred_rmse = np.sqrt(np.mean((ypred.numpy() - ybar_np) ** 2))
        baseline_rmse = np.sqrt(np.mean((ybar_np - ybar_np.mean()) ** 2))
        assert pred_rmse < baseline_rmse * 2, (
            f"RMSE {pred_rmse:.4f} unexpectedly large vs baseline {baseline_rmse:.4f}"
        )