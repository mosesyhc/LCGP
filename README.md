# [Latent component Gaussian process (LCGP)](https://github.com/mosesyhc/LCGP)

[![CI](https://github.com/mosesyhc/lcgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mosesyhc/LCGP/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/mosesyhc/LCGP/badge.svg)](https://coveralls.io/github/mosesyhc/LCGP)
[![Documentation Status](https://readthedocs.org/projects/lcgp/badge/?version=latest)](https://lcgp.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/lcgp.svg)](https://badge.fury.io/py/lcgp)

Implementation of latent component Gaussian process (LCGP).  LCGP handles the emulation
of multivariate stochastic simulation outputs. 

## Reference
The development of this work is described fully in the following work, cited as:
```
@phdthesis{citekey,
  author  = "Moses Y.-H, Chan",
  title   = "High-Dimensional Gaussian Process Methods for Uncertainty Quantification",
  school  = "Northwestern University",
  year    = "2023",
}
```

___

List of Contents:

- [Installation](#installation)
- [Basic Usage](#basic-usage)
  - [What most of us need](#what-most-of-us-need)
  - [Specifying number of latent components](#specifying-number-of-latent-components)
  - [Specifying diagonal error groups](#specifying-diagonal-error-groupings)
  - [Calling different submethod](#define-lcgp-using-different-submethod)
  - [Standardization choices](#standardization-choices)


## Installation
The implementation of LCGP can be installed through

```bash
pip install lcgp
```

> **Note on LBFGS optimizer:** 
> 
> It is strongly recommended that 
> [PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS) is installed to fully utilize
> this implementation.
> Installation guide on PyTorch-LBFGS can be found on 
> [its repository](https://github.com/hjmshi/PyTorch-LBFGS).
> Note that PyTorch-LBFGS has an additional requirement `matplotlib`.  The source code of a version
> of PyTorch-LBFGS that **does not** require `matplotlib` is included in [reference_code](reference_code/).
> 

## Basic usage
### What most of us need:
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
rmse = evaluation.rmse(y, p[0].numpy())
dss = evaluation.dss(y, p[0].numpy(), p[1].numpy(), use_diag=True)
print('Root mean squared error: {:.3E}'.format(rmse))
print('Dawid-Sebastiani score: {:.3f}'.format(dss))

# Access parameters
print(model)
```

### Specifying number of latent components
There are two ways to specify the number of latent components by 
passing one of the following arguments in initializing an LCGP instance:

- `q = 5`: Five latent components will be used.  `q` must be less than or 
equal to the output dimension.
- `var_threshold = 0.99`: Include $q$ latent components such that 99% of the output 
variance are explained, using a singular value decomposition.

> **Note**: Only one of the options should be provided at a time.

```python
model_q = LCGP(y=y, x=x, q=5)
model_var = LCGP(y=y, x=x, var_threshold=0.99)
```

### Specifying diagonal error groupings
If errors of multiple output dimensions are expected to be similar, the error variances
can be grouped in estimation.  

For example, the 6-dimensional output is split into two groups: the
first two have low errors and the remaining four have high errors.

```python
import numpy as np

x = np.linspace(0, 1, 100)
y = np.row_stack((
    np.sin(x), np.cos(x), np.tan(x),
    np.sin(x/2), np.cos(x/2), np.tan(x/2)
))

y[:2] += np.random.normal(2, 1e-3, size=(2, 100))
y[2:] += np.random.normal(-2, 1e-1, size=(4, 100))
```

Then, LCGP can be defined with the argument `diag_error_structure` as a list
of output dimensions to group.  The following code groups the first 2 and the remaining 
4 output dimensions.
```python
model_diag = LCGP(y=y, x=x, diag_error_structure=[2, 4])
```

By default, LCGP assigns a separate error variance to each dimension,
equivalent to 

```python
model_diag = LCGP(y=y, x=x, diag_error_structure=[1]*6)
```

### Define LCGP using different submethod
Three submethods are implemented under LCGP:

* Full posterior (`full`)
* ELBO (`elbo`)
* Profile likelihood (`proflik`)

Under circumstances where the simulation outputs are stochastic, the full posterior 
approach should perform the best.  If the simulation outputs are deterministic, the
profile likelihood method should suffice. 

```python
LCGP_models = []
submethods = ['full', 'elbo', 'proflik']
for submethod in submethods:
    model = LCGP(y=y, x=x, submethod=submethod)
    LCGP_models.append(model)
```

### Standardization choices
LCGP standardizes the simulation output by each dimension to facilitate hyperparameter
training.  The two choices are implemented through `robust_mean = True` or 
`robust_mean = False`. 

* `robust_mean = False`: The empirical mean and standard deviation are used.
* `robust_mean = True`: The empirical median and median absolute error are used.

```python
model = LCGP(y=y, x=x, robust_mean=False)
```

---
