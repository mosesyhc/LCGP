import pytest
import numpy as np
from lcgp.covmat import Matern32


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

