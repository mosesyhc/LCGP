from .lcgp import LCGP
from .covmat import Matern32
from importlib.metadata import version, PackageNotFoundError
from .test import test


try:
    __version__ = version("lcgp")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ['LCGP', 'Matern32', 'test'] # 'parameter_clamping', , 'optim_lbfgs']

__author__ = 'Moses Y.-H. Chan'
__credits__ = 'Northwestern University'