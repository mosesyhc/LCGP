Latent component Gaussian process (LCGP)
========================================

|CI| |Coverage Status| |Documentation Status|

Implementation of latent component Gaussian process (LCGP). LCGP handles
the emulation of multivariate stochastic simulation outputs.

Reference
---------

The development of this work is described fully in the following work, cited as:

.. code-block:: 

   @phdthesis{chan2023thesis,
     author  = "Moses Y.-H. Chan",
     title   = "High-Dimensional Gaussian Process Methods for Uncertainty Quantification",
     school  = "Northwestern University",
     year    = "2023",
   }

--------------

List of Contents:

-  `Installation <#installation>`__
-  `Basic Usage <#basic-usage>`__

   -  `What most of us need <#what-most-of-us-need>`__
   -  `Specifying number of latent
      components <#specifying-number-of-latent-components>`__
   -  `Specifying diagonal error
      groups <#specifying-diagonal-error-groupings>`__
   -  `Calling different
      submethod <#define-lcgp-using-different-submethod>`__
   -  `Standardization choices <#standardization-choices>`__

Installation
------------

The implementation of LCGP requires Python 3.9 or above.  The package can be installed through

.. code:: bash

   pip install lcgp

..

The LCGP package has the following dependencies, as listed in its pyproject.toml configuration:

.. code:: python

    'numpy>=1.18.3',
    'scipy>=1.10.1',
    'tensorflow>=2.16.0',
    'gpflow>=2.5.0'

..

Test suite
~~~~~~~~~~

A list of basic tests is provided for user to verify that LCGP is installed correctly.
Execute the follow code within the root directory:

.. code:: bash

    $ python
    >>> import lcgp
    >>> lcgp.__version__
    <version string>
    >>> lcgp.test()

Basic usage
-----------

What most of us need:
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

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

Specifying number of latent components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to specify the number of latent components by passing
one of the following arguments in initializing an LCGP instance:

-  ``q = 5``: Five latent components will be used. ``q`` must be less
   than or equal to the output dimension.
-  ``var_threshold = 0.99``: Include :math:`q` latent components such
   that 99% of the output variance are explained, using a singular value
   decomposition.

..

   **Note**: Only one of the options should be provided at a time.

.. code:: python

   model_q = LCGP(y=y, x=x, q=5)
   model_var = LCGP(y=y, x=x, var_threshold=0.99)

Specifying diagonal error groupings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If errors of multiple output dimensions are expected to be similar, the
error variances can be grouped in estimation.

For example, the 6-dimensional output is split into two groups: the
first two have low errors and the remaining four have high errors.

.. code:: python

   import numpy as np

   x = np.linspace(0, 1, 100)
   y = np.row_stack((
       np.sin(x), np.cos(x), np.tan(x),
       np.sin(x/2), np.cos(x/2), np.tan(x/2)
   ))

   y[:2] += np.random.normal(2, 1e-3, size=(2, 100))
   y[2:] += np.random.normal(-2, 1e-1, size=(4, 100))

Then, LCGP can be defined with the argument ``diag_error_structure`` as
a list of output dimensions to group. The following code groups the
first 2 and the remaining 4 output dimensions.

.. code:: python

   model_diag = LCGP(y=y, x=x, diag_error_structure=[2, 4])

By default, LCGP assigns a separate error variance to each dimension,
equivalent to

.. code:: python

   model_diag = LCGP(y=y, x=x, diag_error_structure=[1]*6)

Define LCGP using different submethod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main and recommended method under LCGP is the Full posterior (``full``) method.  The method
takes into account the uncertainty propagated to the latent components and integrates out
the latent components.

In the current release, the full posterior method is fully implemented under LCGP, whereas ELBO
(``elbo``) and Profile likelihood (``proflik``) methods are under reconstruction.

Under circumstances where the simulation outputs are stochastic, the
full posterior approach should perform the best. If the simulation
outputs are deterministic, the profile likelihood method should suffice.

.. code:: python

   LCGP_models = []
   submethods = ['full', 'elbo', 'proflik']
   for submethod in submethods:
       model = LCGP(y=y, x=x, submethod=submethod)
       LCGP_models.append(model)

Standardization choices
~~~~~~~~~~~~~~~~~~~~~~~

LCGP standardizes the simulation output by each dimension to facilitate
hyperparameter training. The two choices are implemented through
``robust_mean = True`` or ``robust_mean = False``.

-  ``robust_mean = False``: The empirical mean and standard deviation
   are used.
-  ``robust_mean = True``: The empirical median and median absolute
   error are used.

.. code:: python

   model = LCGP(y=y, x=x, robust_mean=False)

--------------

.. |CI| image:: https://github.com/mosesyhc/lcgp/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/mosesyhc/LCGP/actions/workflows/ci.yml
.. |Coverage Status| image:: https://coveralls.io/repos/github/mosesyhc/LCGP/badge.svg
   :target: https://coveralls.io/github/mosesyhc/LCGP
.. |Documentation Status| image:: https://readthedocs.org/projects/lcgp/badge/?version=latest
   :target: https://lcgp.readthedocs.io/en/latest/?badge=latest
