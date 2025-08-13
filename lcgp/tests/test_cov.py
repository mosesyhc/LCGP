import pytest
import numpy as np
import copy

from lcgp import LCGP
from lcgp.covmat import Matern32


class TestInit:
    def test_simplest_1D_fail(self):
        x = np.linspace(0, 1, 40)
        y = copy.copy(x)
        with pytest.raises(AssertionError):
            LCGP(y=y, x=x)

    def test_simplest_1D_pass(self):
        x = np.linspace(0, 1, 40)
        y = np.reshape(copy.copy(x), (1, 40))
        LCGP(y=y, x=x)

    def test_simplest_HD(self):
        x = np.random.randn(40, 5)
        y = np.random.randn(3, 40)
        LCGP(y=y, x=x)

    def test_print_model(self):
        x = np.random.randn(40, 5)
        y = np.random.randn(3, 40)
        model = LCGP(y=y, x=x)
        print(model)

    @pytest.mark.parametrize('err_struct', [[2, 1], [1, 1, 1], None, [1, 2]])
    def test_err_struct(self, err_struct):
        x = np.random.randn(40, 5)
        y = np.random.randn(3, 40)
        LCGP(y=y, x=x, diag_error_structure=err_struct)

    @pytest.mark.parametrize('err_struct', [[1, 1], [0, 1, 1], [2, 2]])
    def test_invalid_err_struct(self, err_struct):
        x = np.random.randn(40, 5)
        y = np.random.randn(3, 40)
        with pytest.raises(AssertionError):
            LCGP(y=y, x=x, diag_error_structure=err_struct)

    @pytest.mark.parametrize('robust_mean', [True, False])
    def test_robust(self, robust_mean):
        x = np.linspace(0, 1, 40)
        y = np.reshape(copy.copy(x), (1, 40))
        LCGP(y=y, x=x, robust_mean=robust_mean)

    def test_invalid_q_varthreshold(self):
        x = np.linspace(0, 1, 40)
        y = np.random.randn(3, 40)
        with pytest.raises(ValueError):
            LCGP(y=y, x=x, q=2, var_threshold=0.9)

    def test_varthreshold(self):
        x = np.linspace(0, 1, 40)
        y = np.random.randn(3, 40)
        LCGP(y=y, x=x, q=None, var_threshold=0.9)

    @pytest.mark.parametrize('penalty_constant', [None,
                                                  {'lLmb': 0, 'lLmb0': 0},
                                                  {'lLmb': 40, 'lLmb0': 5}])
    def test_penalty(self, penalty_constant):
        x = np.linspace(0, 1, 40)
        y = np.reshape(copy.copy(x), (1, 40))
        LCGP(y=y, x=x, penalty_const=penalty_constant)

    @pytest.mark.parametrize('penalty_constant', [{'lLmb': -5, 'lLmb0': 10},
                                                  {'lLmb': 5, 'lLmb0': -10}])
    def test_invalid_penalty(self, penalty_constant):
        x = np.linspace(0, 1, 40)
        y = np.reshape(copy.copy(x), (1, 40))
        with pytest.raises(AssertionError):
            LCGP(y=y, x=x, penalty_const=penalty_constant)

    @pytest.mark.parametrize('x, y', [(np.linspace(0, 1, 40),
                                       np.random.randn(3, 25))])
    def test_mismatch_dimension(self, x, y):
        with pytest.raises(AssertionError):
            LCGP(y=y, x=x)

    def test_tx_xy(self):
        x = np.linspace(0, 1, 40)
        y = np.reshape(copy.copy(x), (1, 40))
        model = LCGP(y=y, x=x)
        model.tx_x(model.x)
        model.tx_y(model.y)

@pytest.mark.parametrize('llmb, llmb0, lnug', [((1.), (1.), (-12.))])
class TestCov1D:
    @pytest.mark.parametrize('x1', [np.reshape((np.linspace(0, 1, 40)), (40, 1))])
    @pytest.mark.parametrize('x2', [np.reshape(np.linspace(0, 1, 40), (40, 1)),
                                    np.reshape(np.linspace(0, 1, 25), (25, 1))])
    def test_Matern1D(self, x1, x2, llmb, llmb0, lnug):
        Matern32(x1=x1, x2=x2, llmb=llmb, llmb0 = llmb0, lnug=lnug)

    @pytest.mark.parametrize('x1', [np.reshape(np.linspace(0, 1, 40), (40, 1))])
    def test_Matern1D_diag(self, x1, llmb, llmb0, lnug):
        Matern32(x1, x1, llmb=llmb, llmb0=llmb0, lnug=lnug, diag_only=True)

    @pytest.mark.parametrize('x1', [np.linspace(0, 1, 40)])
    @pytest.mark.parametrize('x2', [np.linspace(0, 1, 40),
                                    np.linspace(0, 1, 25)])
    def test_invalid_Matern1D(self, x1, x2, llmb, llmb0, lnug):
        with pytest.raises(AssertionError):
            Matern32(x1=x1, x2=x2, llmb=llmb, llmb0=llmb0, lnug=lnug)
    
    # @pytest.mark.parametrize('x', [np.linspace(0, 1, 40).unsqueeze(1)])
    # @pytest.mark.parametrize('xi', [np.linspace(0, 1, 40).unsqueeze(1),
    #                                 np.linspace(0, 1, 25).unsqueeze(1)])
    # def test_Matern1D_sp(self, x, xi, llmb, llmb0, lnug):
    #     Matern32_sp(x=x, xi=xi, llmb=llmb, llmb0 = llmb0, lnug=lnug)
    # 
    # @pytest.mark.parametrize('x', [np.linspace(0, 1, 40)])
    # @pytest.mark.parametrize('xi', [np.linspace(0, 1, 40),
    #                                 np.linspace(0, 1, 25)])
    # def test_invalid_Matern1D_sp(self, x, xi, llmb, llmb0, lnug):
    #     with pytest.raises(AssertionError):
    #         Matern32_sp(x=x, xi=xi, llmb=llmb, llmb0=llmb0, lnug=lnug)

@pytest.mark.parametrize('llmb, llmb0, lnug', [(([1., 1.]), (1.),
                                                (-12.))])
class TestCovHD:
    @pytest.mark.parametrize('X1, X2', [(np.random.randn(40, 2),
                                         np.random.randn(40, 2))])
    def test_Matern2D(self, X1, X2, llmb, llmb0, lnug):
        Matern32(x1=X1, x2=X2, llmb=llmb, llmb0=llmb0, lnug=lnug)

    @pytest.mark.parametrize('X1', [np.random.randn(40, 2)])
    def test_Matern2D_diag(self, X1, llmb, llmb0, lnug):
        Matern32(x1=X1, x2=X1, llmb=llmb, llmb0=llmb0, lnug=lnug, diag_only=True)

