from importlib.metadata import PackageNotFoundError, version

from .covmat import Matern32
from .lcgp import LCGP
from .test import test

try:
    __version__ = version("lcgp")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ['LCGP', 'Matern32', 'test'] # 'parameter_clamping', , 'optim_lbfgs']

__author__ = 'Moses Y.-H. Chan'
__credits__ = 'Northwestern University'