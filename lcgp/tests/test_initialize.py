import pytest
import numpy as np
import copy

from lcgp import LCGP


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
