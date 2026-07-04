import pytest
import numpy as np

from lcgp.evaluation import dss, rmse, normalized_rmse, intervalstats


@pytest.mark.parametrize('y1', [np.random.normal(0, 1, (3, 40))])
class TestEvaluation:
    def test_rmse(self, y1):
        y2 = y1 + 1e-16 * np.random.randn(*y1.shape)
        assert np.isclose(rmse(y1, y2), 0.)

    def test_normalized_rmse(self, y1):
        y2 = y1 + 1e-16 * np.random.randn(*y1.shape)
        assert np.isclose(normalized_rmse(y1, y2), 0.)

    def test_dss(self, y1):
        mean = np.zeros_like(y1)
        p, n = y1.shape
        cov = np.repeat(np.eye(p), n).reshape((p, p, n))
        dss(y=y1, ypredmean=mean, ypredcov=cov, use_diag=False)

    def test_dss_diag(self, y1):
        mean = np.zeros_like(y1)
        var = np.ones_like(y1)
        dss(y=y1, ypredmean=mean, ypredcov=var, use_diag=True)

    def test_intervals(self, y1):
        mean = np.zeros_like(y1)
        var = np.ones_like(y1)
        coverage, length = intervalstats(y1, mean, var)
        assert 0 <= coverage <= 1
        assert length >= 0

