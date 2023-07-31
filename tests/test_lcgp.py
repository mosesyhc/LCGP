import pytest
import numpy as np
import torch
from torch import tensor

from lcgp import LCGP
from lcgp.evaluation import dss, rmse, normalized_rmse, intervalstats
from lcgp.covmat import Matern32, Matern32_sp
from lcgp.hyperparameter_tuning import parameter_clamping


class TestInit:
    @pytest.mark.parametrize('robust_mean', [True, False])
    def test_robust(self, robust_mean):
        x = torch.linspace(0, 1, 100)
        y = x.clone()
        LCGP(y=y, x=x, robust_mean=robust_mean)

    def test_invalid_q_varthreshold(self):
        x = torch.linspace(0, 1, 100)
        y = torch.randn((3, 100))
        with pytest.raises(ValueError):
            LCGP(y=y, x=x, q=2, var_threshold=0.9)

    def test_varthreshold(self):
        x = torch.linspace(0, 1, 100)
        y = torch.randn((3, 100))
        LCGP(y=y, x=x, var_threshold=0.9)

    @pytest.mark.parametrize('penalty_constant', [None,
                                                  {'lLmb': 0, 'lLmb0': 0},
                                                  {'lLmb': 40, 'lLmb0': 5}])
    def test_penalty(self, penalty_constant):
        x = torch.linspace(0, 1, 100)
        y = x.clone()
        LCGP(y=y, x=x, penalty_const=penalty_constant)

    @pytest.mark.parametrize('penalty_constant', [{'lLmb': -5, 'lLmb0': 10},
                                                  {'lLmb': 5, 'lLmb0': -10}])
    def test_invalid_penalty(self, penalty_constant):
        x = torch.linspace(0, 1, 100)
        y = x.clone()
        with pytest.raises(AssertionError):
            LCGP(y=y, x=x, penalty_const=penalty_constant)

    @pytest.mark.parametrize('x, y', [(torch.linspace(0, 1, 100), torch.randn((3, 50)))])
    def test_mismatch_dimension(self, x, y):
        with pytest.raises(AssertionError):
            LCGP(y=y, x=x)

    def test_tx_xy(self):
        x = torch.linspace(0, 1, 100)
        y = x.clone()
        model = LCGP(y=y, x=x)
        model.tx_x(model.x)
        model.tx_y(model.y)

@pytest.mark.parametrize('llmb, llmb0, lnug', [(tensor(0.), tensor(0.), tensor(-12.))])
class TestCov1D:
    @pytest.mark.parametrize('x1', [torch.linspace(0, 1, 100).unsqueeze(1)])
    @pytest.mark.parametrize('x2', [torch.linspace(0, 1, 100).unsqueeze(1),
                                    torch.linspace(0, 1, 50).unsqueeze(1)])
    def test_Matern1D(self, x1, x2, llmb, llmb0, lnug):
        Matern32(x1=x1, x2=x2, llmb=llmb, llmb0 = llmb0, lnug=lnug)

    @pytest.mark.parametrize('x1', [torch.linspace(0, 1, 100).unsqueeze(1)])
    def test_Matern1D_diag(self, x1, llmb, llmb0, lnug):
        Matern32(x1, x1, llmb=llmb, llmb0=llmb0, lnug=lnug, diag_only=True)

    @pytest.mark.parametrize('x1', [torch.linspace(0, 1, 100)])
    @pytest.mark.parametrize('x2', [torch.linspace(0, 1, 100),
                                    torch.linspace(0, 1, 50)])
    def test_invalid_Matern1D(self, x1, x2, llmb, llmb0, lnug):
        with pytest.raises(AssertionError):
            Matern32(x1=x1, x2=x2, llmb=llmb, llmb0=llmb0, lnug=lnug)

    @pytest.mark.parametrize('x', [torch.linspace(0, 1, 100).unsqueeze(1)])
    @pytest.mark.parametrize('xi', [torch.linspace(0, 1, 100).unsqueeze(1),
                                    torch.linspace(0, 1, 50).unsqueeze(1)])
    def test_Matern1D_sp(self, x, xi, llmb, llmb0, lnug):
        Matern32_sp(x=x, xi=xi, llmb=llmb, llmb0 = llmb0, lnug=lnug)

    @pytest.mark.parametrize('x', [torch.linspace(0, 1, 100)])
    @pytest.mark.parametrize('xi', [torch.linspace(0, 1, 100),
                                    torch.linspace(0, 1, 50)])
    def test_invalid_Matern1D_sp(self, x, xi, llmb, llmb0, lnug):
        with pytest.raises(AssertionError):
            Matern32_sp(x=x, xi=xi, llmb=llmb, llmb0=llmb0, lnug=lnug)

@pytest.mark.parametrize('llmb, llmb0, lnug', [(tensor([0., 0.]), tensor(0.), tensor(-12.))])
class TestCovHD:
    @pytest.mark.parametrize('X1, X2', [(torch.randn((100, 2)),
                                         torch.randn((100, 2)))])
    def test_Matern2D(self, X1, X2, llmb, llmb0, lnug):
        Matern32(x1=X1, x2=X2, llmb=llmb, llmb0=llmb0, lnug=lnug)

    @pytest.mark.parametrize('X1', [torch.randn((100, 2))])
    def test_Matern2D_diag(self, X1, llmb, llmb0, lnug):
        Matern32(x1=X1, x2=X1, llmb=llmb, llmb0=llmb0, lnug=lnug, diag_only=True)


@pytest.mark.parametrize('y1', [np.random.normal(0, 1, (3, 100))])
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


class TestTraining:
    def test_fit(self):
        x = torch.linspace(0, 1, 100)
        y = x.clone()
        model = LCGP(y=y, x=x)
        model.fit(maxiter=15)
        model.predict(x0=x)
        model.predict(x0=x, return_fullcov=True)
        model.forward(x0=x)
        model.get_param()
        model.get_param_grad()

    def test_optim_edge_lr(self):
        x = torch.linspace(0, 1, 100)
        y = x.clone()
        model = LCGP(y=y, x=x)
        model.fit(maxiter=15, lr=20, verbose=True)

    def test_optim_edge_ftol(self):
        x = torch.linspace(0, 1, 100)
        y = x.clone()
        model = LCGP(y=y, x=x)
        model.fit(maxiter=15, ftol=1, verbose=True)

    def test_optim_edge_pgtol(self):
        x = torch.linspace(0, 1, 100)
        y = x.clone()
        model = LCGP(y=y, x=x)
        model.fit(maxiter=15, pgtol=1, verbose=True)


class TestParameterClamping:
    @pytest.mark.parametrize('clamping', [True, False])
    def test_parameter_clamping_1d(self, clamping):
        x = torch.linspace(0, 1, 100)
        y = x.clone()

        model = LCGP(y=y, x=x, parameter_clamp_flag=clamping)
        model.get_param()

    @pytest.mark.parametrize('clamping', [True, False])
    def test_parameter_clamping_hd(self, clamping):
        t = torch.randn(4)
        trng = tensor([(0, 1), (0, 1), (0, 1), (0, 1)])

        parameter_clamping(t=t, trng=trng)
