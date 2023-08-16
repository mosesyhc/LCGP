# Latent component Gaussian process (LCGP)

[![CI](https://github.com/mosesyhc/lcgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mosesyhc/LCGP/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/mosesyhc/LCGP/badge.svg)](https://coveralls.io/github/mosesyhc/LCGP)
[![Documentation Status](https://readthedocs.org/projects/lcgp/badge/?version=latest)](https://lcgp.readthedocs.io/en/latest/?badge=latest)

Implementation of latent component Gaussian process (LCGP).

___

List of Contents:

- [Installation](#installation)
- [Basic Usage](#usage)

## Installation 
The implementation of LCGP can be installed through

```bash
pip install lcgp
```

## Basic usage
```python
import numpy as np
from lcgp import LCGP
from lcgp import evaluation  # optional evaluation module

# Generate fifty 2-dimensional input and 4-dimensional output
x = np.random.randn(50, 2)
y = np.random.randn(4, 50)

# Define LCGP model
model = LCGP(y=y, x=x)

# Estimate error covariance and hyperparameters
model.fit()

# Prediction
p = model.predict(x0=x)  # mean and variance
rmse = evaluation.rmse(y, p[0])
dss = evaluation.dss(y, *p, use_diag=True)
print('Root mean squared error: {:.3E}'.format(rmse))
print('Dawid-Sebastiani score: {:.3f}'.format(dss))

# Access parameters
model.get_param()
```

