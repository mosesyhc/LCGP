import pytest
import numpy as np
import tensorflow as tf
from lcgp import LCGP


def _make_rep_data(seed=0, n_unique=20, p=4, d=2, reps=3):
    rng = np.random.default_rng(seed)
    x_unique = rng.uniform(0, 1, (n_unique, d))
    x = np.tile(x_unique, (reps, 1))
    y = rng.standard_normal((p, n_unique * reps))
    return x, y, x_unique, n_unique


def _make_full_data(seed=0, n=50, p=4, d=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, d))
    y = rng.standard_normal((p, n))
    return x, y

class TestComputeCenterSpreadNonRobust:
    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_center_spread_non_robust(self, n_unique, reps, p, d):
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep', robust_mean=False)
        Y = model.ybar
        ycenter, yspread = model._compute_center_spread_tf(Y)

        expected_center = tf.reduce_mean(Y, axis=1, keepdims=True)
        expected_spread = tf.math.reduce_std(Y, axis=1, keepdims=True)

        np.testing.assert_allclose(ycenter.numpy(), expected_center.numpy())
        np.testing.assert_allclose(yspread.numpy(), expected_spread.numpy())

class TestPreprocess:
    @pytest.mark.parametrize('n_unique,reps,p,d', [(15, 4, 3, 2)])
    def test_preprocess_returns_expected_tuple(self, n_unique, reps, p, d):
        x, y, x_unique, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')

        result = model.preprocess(x_raw=x, y_raw=y)
        (x_unique_tf, x_unique_s, group_ids_tf, r_tf, R_tf,
         ybar_tf, ybar_s_tf, ybar_mean_tf, ybar_std_tf,
         n_unique_out, d_out, p_out) = result

        assert int(n_unique_out.numpy()) == n_unique
        assert int(d_out.numpy()) == d
        assert int(p_out.numpy()) == p
        assert x_unique_tf.shape == (n_unique, d)
        assert ybar_tf.shape == (p, n_unique)
        assert ybar_s_tf.shape == (p, n_unique)
        assert R_tf.shape == (n_unique, n_unique)

    @pytest.mark.parametrize('n_unique,reps,p,d', [(15, 4, 3, 2)])
    def test_preprocess_default_raw_args(self, n_unique, reps, p, d):
        """Calling preprocess() with no args resolves x_raw/y_raw from self.x_orig/y_orig."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')

        result = model.preprocess()
        assert result[9].numpy() == n_unique  # n_unique
        assert result[10].numpy() == d         # d
        assert result[11].numpy() == p         # p

class TestEnsureReplication:
    @pytest.mark.parametrize('n_unique,reps,p,d', [(15, 4, 3, 2)])
    def test_ensure_replication_already_initialized(self, n_unique, reps, p, d):
        """When _rep_initialized is True, preprocess() should NOT be called again."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        assert model._rep_initialized is True

        # spy on preprocess
        called = {'count': 0}
        original_preprocess = model.preprocess

        def spy_preprocess(*args, **kwargs):
            called['count'] += 1
            return original_preprocess(*args, **kwargs)

        model.preprocess = spy_preprocess
        model._ensure_replication()
        assert called['count'] == 0

    @pytest.mark.parametrize('n_unique,reps,p,d', [(15, 4, 3, 2)])
    def test_ensure_replication_triggers_preprocess(self, n_unique, reps, p, d):
        """When _rep_initialized is False, preprocess() should be called."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')

        # force flag back to False
        model._rep_initialized = False

        called = {'count': 0}
        original_preprocess = model.preprocess

        def spy_preprocess(*args, **kwargs):
            called['count'] += 1
            return original_preprocess(*args, **kwargs)

        model.preprocess = spy_preprocess
        model._ensure_replication()
        assert called['count'] == 1
        assert model._rep_initialized is True

class TestGetPhiInputFallback:
    @pytest.mark.parametrize('n_unique,reps,p,d', [(15, 4, 3, 2)])
    def test_returns_ybar_when_std_disabled(self, n_unique, reps, p, d):
        """rep_standardize_ybar=False -> _get_phi_input should fall back to self.ybar."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep', rep_standardize_ybar=False)

        out = model._get_phi_input()
        np.testing.assert_allclose(out.numpy(), model.ybar.numpy())

    @pytest.mark.parametrize('n_unique,reps,p,d', [(15, 4, 3, 2)])
    def test_returns_y_when_no_ybar(self, n_unique, reps, p, d):
        """If neither ybar_s nor ybar exist, fall back to self.y."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')

        del model.ybar_s
        del model.ybar

        out = model._get_phi_input()
        np.testing.assert_allclose(out.numpy(), model.y.numpy())

class TestNeglpostRepNonStd:
    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_neglpost_rep_non_standardized(self, n_unique, reps, p, d):
        """rep_standardize_ybar=False path: ybar = self.ybar, sigma_var_used = sigma_var_raw."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep', rep_standardize_ybar=False)

        loss = model.neglpost_rep()
        assert np.isfinite(float(loss))

class TestPredictInvalidSubmethod:
    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_predict_raises_keyerror_for_bad_submethod(self, n_unique, reps, p, d):
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')

        model.submethod = 'bogus'

        with pytest.raises(KeyError):
            model.predict(x0=model.x_unique)

class TestComputeAuxDispatch:
    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_compute_aux_predictive_quantities_dispatches_to_rep(self, n_unique, reps, p, d):
        """
        compute_aux_predictive_quantities() should detect x_unique/ybar attrs
        and delegate to _compute_aux_predictive_quantities_rep, returning early.
        """
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')

        # reset to nan to force recomputation
        model.CinvMs = tf.fill([model.q, model.n], tf.constant(float('nan'), dtype=tf.float64))
        model.Tks = None

        model.compute_aux_predictive_quantities()

        assert model.Tks is not None
        assert not np.any(np.isnan(model.CinvMs.numpy()))


class TestPredictFullFullCov:
    @pytest.mark.parametrize('n,p,d', [(40, 3, 2)])
    def test_predict_full_return_fullcov(self, n, p, d):
        x, y = _make_full_data(n=n, p=p, d=d)
        model = LCGP(y=y, x=x, submethod='full')
        model.fit()

        n0 = 8
        x0 = np.random.default_rng(11).uniform(0, 1, (n0, d))
        ypred, ypredvar, yconfvar, yfullpredcov = model.predict(x0, return_fullcov=True)

        assert ypred.shape == (p, n0)
        assert ypredvar.shape == (p, n0)
        assert yconfvar.shape == (p, n0)
        assert yfullpredcov.shape == (n0, p, p)
        assert np.all(np.isfinite(yfullpredcov.numpy()))

        diag = np.diagonal(yfullpredcov.numpy(), axis1=1, axis2=2).T  # (p, n0)
        np.testing.assert_allclose(diag, ypredvar.numpy(), rtol=1e-5, atol=1e-6)


class TestPredictRepNonStdAndFullCov:
    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_predict_rep_non_standardized(self, n_unique, reps, p, d):
        """rep_standardize_ybar=False path in predict_rep."""
        x, y, x_unique, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep', rep_standardize_ybar=False)
        model.fit()

        x0 = np.random.default_rng(12).uniform(0, 1, (10, d))
        ypred, ypredvar, yconfvar = model.predict(x0)

        assert ypred.shape == (p, 10)
        assert np.all(np.isfinite(ypred.numpy()))
        assert np.all(np.isfinite(ypredvar.numpy()))
        assert np.all(np.isfinite(yconfvar.numpy()))

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_predict_rep_return_fullcov(self, n_unique, reps, p, d):
        """return_fullcov=True branch returns (ypred, ypredvar, yconfvar, None)."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep')
        model.fit()

        x0 = np.random.default_rng(13).uniform(0, 1, (10, d))
        ypred, ypredvar, yconfvar, fullcov = model.predict(x0, return_fullcov=True)

        assert fullcov is None
        assert ypred.shape == (p, 10)
        assert np.all(np.isfinite(ypred.numpy()))

    @pytest.mark.parametrize('n_unique,reps,p,d', [(20, 3, 4, 2)])
    def test_predict_rep_non_standardized_return_fullcov(self, n_unique, reps, p, d):
        """Combine both: rep_standardize_ybar=False AND return_fullcov=True."""
        x, y, _, _ = _make_rep_data(n_unique=n_unique, p=p, d=d, reps=reps)
        model = LCGP(y=y, x=x, submethod='rep', rep_standardize_ybar=False)
        model.fit()

        x0 = np.random.default_rng(14).uniform(0, 1, (10, d))
        ypred, ypredvar, yconfvar, fullcov = model.predict(x0, return_fullcov=True)

        assert fullcov is None
        assert np.all(np.isfinite(ypred.numpy()))
        assert np.all(np.isfinite(ypredvar.numpy()))
        assert np.all(np.isfinite(yconfvar.numpy()))