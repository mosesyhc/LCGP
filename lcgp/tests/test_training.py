import pytest
import numpy as np
import copy

from lcgp import LCGP


class TestTraining:
    @pytest.mark.parametrize('submethod', ['full'])  #, 'elbo', 'proflik'])
    def test_fit(self, submethod):
        x = np.linspace(0, 1, 40)
        y = np.reshape(copy.copy(x), (1, 40))
        model = LCGP(y=y, x=x, submethod=submethod)
        model.fit()
        model.predict(x0=x)
        # model.predict(x0=x, return_fullcov=True)
        # model.forward(x0=x)
        model.get_param()
        # model.get_param_grad()

    def test_invalid_submethod(self):
        x = np.linspace(0, 1, 40)
        y = np.reshape(copy.copy(x), (1, 40))
        model = LCGP(y=y, x=x, submethod='null')
        with pytest.raises(ValueError):
            model.fit()
            model.predict(x0=x)
