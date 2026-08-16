"""
Minimal in-house replacements for the GPflow and TensorFlow Probability
functionality LCGP uses.

Motivation: GPflow declares a dependency on the retired ``tensorflow-macos``
package for Apple Silicon and caps ``numpy<2``, and TensorFlow Probability
imports ``distutils`` in the module. Both make the dependency tree
unresolvable on recent Python versions and unusable on Apple Silicon. 
Since only TensorFlow, NumPy and SciPy are required for the functionality, 
this file replaces the used GPflow dependencies.

Public names mirror the semantics of the pieces they replace:

    SoftClip                  <- tfp.bijectors.SoftClip
    percentile                <- tfp.stats.percentile (interpolation='nearest')
    Module                    <- gpflow.Module
    Parameter                 <- gpflow.Parameter
    scipy_minimize            <- gpflow.optimizers.Scipy().minimize
    tabulate_module_summary   <- gpflow.utilities.tabulate_module_summary
"""

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize as _scipy_minimize

__all__ = [
    "Bijector",
    "Identity",
    "Module",
    "Parameter",
    "SoftClip",
    "percentile",
    "scipy_minimize",
    "tabulate_module_summary",
]


# ---------------------------------------------------------------------------
# Bijectors
# ---------------------------------------------------------------------------
def _softplus(x):
    return tf.math.softplus(x)


def _softplus_inverse(y):
    """Stable log(exp(y) - 1)."""
    return y + tf.math.log(-tf.math.expm1(-y))


class Bijector:
    def forward(self, x):
        raise NotImplementedError

    def inverse(self, y):
        raise NotImplementedError


class Identity(Bijector):
    def forward(self, x):
        return tf.convert_to_tensor(x)

    def inverse(self, y):
        return tf.convert_to_tensor(y)


class SoftClip(Bijector):
    """
    Smooth map from the reals onto ``(low, high)``.

    Same construction as ``tfp.bijectors.SoftClip`` with both bounds set and
    ``hinge_softness=1``:

        forward(x) = high - softplus(w - softplus(x - low)) * w / softplus(w)

    with ``w = high - low``. Away from the bounds the map is close to the
    identity; near them it saturates smoothly, which is what keeps the
    optimizer inside the feasible region.
    """

    def __init__(self, low, high, dtype=tf.float64, name="soft_clip"):
        self.low = tf.convert_to_tensor(low, dtype=dtype)
        self.high = tf.convert_to_tensor(high, dtype=dtype)
        self.dtype = dtype
        self.name = name
        self._width = self.high - self.low
        self._scale = self._width / _softplus(self._width)

    def forward(self, x):
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        inner = _softplus(self._width - _softplus(x - self.low))
        return self.high - inner * self._scale

    def inverse(self, y):
        y = tf.convert_to_tensor(y, dtype=self.dtype)
        inner = _softplus_inverse((self.high - y) / self._scale)
        return self.low + _softplus_inverse(self._width - inner)

    def __repr__(self):
        return f"SoftClip(low={float(self.low.numpy())}, high={float(self.high.numpy())})"


# ---------------------------------------------------------------------------
# Parameters and modules
# ---------------------------------------------------------------------------
class Module(tf.Module):
    """
    ``tf.Module`` with GPflow's attribute-name behaviour.
    """

    def __init__(self, name=None):
        super().__init__(name=None)


class Parameter(tf.Module):
    """
    A trainable quantity stored in unconstrained space.

    The variable exposed to the optimizer through ``trainable_variables`` is the
    unconstrained one. Reading the parameter (in a TF op, via ``__getitem__``,
    ``numpy()`` or NumPy coercion) applies the forward transform.
    """

    def __init__(self, value, transform=None, name=None, dtype=tf.float64,
                 trainable=True):
        super().__init__(name=None)
        self.param_name = name
        self.transform = transform if transform is not None else Identity()
        self.param_dtype = dtype
        constrained = tf.convert_to_tensor(value, dtype=dtype)
        self._unconstrained = tf.Variable(
            self.transform.inverse(constrained),
            dtype=dtype,
            trainable=trainable,
        )

    # -- value access -------------------------------------------------------
    @property
    def unconstrained_variable(self):
        return self._unconstrained

    def value(self):
        return self.transform.forward(self._unconstrained)

    def assign(self, value):
        """Assign in constrained space, matching gpflow.Parameter.assign."""
        constrained = tf.convert_to_tensor(value, dtype=self.param_dtype)
        self._unconstrained.assign(self.transform.inverse(constrained))
        return self

    # -- tensor protocol ----------------------------------------------------
    def __tf_tensor__(self, dtype=None, name=None):
        t = self.value()
        if dtype is not None and dtype != t.dtype:
            t = tf.cast(t, dtype)
        return t

    def __array__(self, dtype=None, copy=None):
        arr = self.value().numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def numpy(self):
        return self.value().numpy()

    def __getitem__(self, key):
        return self.value()[key]

    def __iter__(self):
        return iter(tf.unstack(self.value()))

    @property
    def shape(self):
        return self._unconstrained.shape

    @property
    def dtype(self):
        return self.param_dtype

    @property
    def ndim(self):
        return len(self._unconstrained.shape)

    @property
    def trainable(self):
        return self._unconstrained.trainable

    def __repr__(self):
        return (f"Parameter({self.param_name}, shape={tuple(self.shape)}, "
                f"transform={self.transform})")

    # -- arithmetic ---------------------------------------------------------
    def _binary(op):
        def f(self, other):
            return op(self.value(), other)
        return f

    def _rbinary(op):
        def f(self, other):
            return op(other, self.value())
        return f

    __add__ = _binary(lambda a, b: a + b)
    __sub__ = _binary(lambda a, b: a - b)
    __mul__ = _binary(lambda a, b: a * b)
    __truediv__ = _binary(lambda a, b: a / b)
    __pow__ = _binary(lambda a, b: a ** b)
    __radd__ = _rbinary(lambda a, b: a + b)
    __rsub__ = _rbinary(lambda a, b: a - b)
    __rmul__ = _rbinary(lambda a, b: a * b)
    __rtruediv__ = _rbinary(lambda a, b: a / b)

    del _binary, _rbinary

    def __neg__(self):
        return -self.value()


def tabulate_module_summary(module, fmt=None):
    """Plain-text summary of the parameters held by ``module``."""
    rows = []
    for path, param in _walk_parameters(module):
        val = param.numpy()
        rows.append((
            path,
            str(param.param_name or ""),
            type(param.transform).__name__,
            str(param.trainable),
            str(tuple(param.shape)),
            str(np.asarray(val).dtype),
            _summarize(val),
        ))
    if not rows:
        return "(no parameters)"
    header = ("name", "label", "transform", "trainable", "shape", "dtype", "value")
    widths = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    return f"{line}\n{sep}\n{body}"


def _walk_parameters(module, prefix=""):
    seen = set()
    for name, attr in vars(module).items():
        if isinstance(attr, Parameter) and id(attr) not in seen:
            seen.add(id(attr))
            yield prefix + name, attr


def _summarize(val):
    arr = np.asarray(val)
    if arr.size == 1:
        return f"{float(arr.reshape(-1)[0]):.6g}"
    flat = arr.reshape(-1)
    head = ", ".join(f"{v:.4g}" for v in flat[:4])
    return f"[{head}{', ...' if flat.size > 4 else ''}]"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def percentile(x, q, axis=None, keepdims=False, interpolation="nearest"):
    """
    ``tfp.stats.percentile`` replacement.

    The default interpolation is ``'nearest'``, matching TFP rather than NumPy
    (whose default is ``'linear'``). 
    """
    t = tf.convert_to_tensor(x)
    arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    out = np.percentile(arr, q, axis=axis, keepdims=keepdims, method=interpolation)
    return tf.convert_to_tensor(out, dtype=t.dtype)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def scipy_minimize(closure, variables, method="L-BFGS-B", options=None,
                   callback=None, **kwargs):
    """
    ``gpflow.optimizers.Scipy().minimize`` replacement.

    Flattens ``variables`` into a single float64 vector, evaluates the loss and
    its gradient with ``tf.GradientTape``, and hands both to
    ``scipy.optimize.minimize``. Returns the SciPy ``OptimizeResult``.
    """
    variables = list(variables)
    shapes = [tuple(v.shape) for v in variables]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]

    def _pack(arrays):
        return np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1)
                               for a in arrays]) if arrays else np.zeros(0)

    def _assign(flat):
        i = 0
        for v, shape, size in zip(variables, shapes, sizes):
            chunk = flat[i:i + size].reshape(shape)
            v.assign(tf.convert_to_tensor(chunk, dtype=v.dtype))
            i += size

    def _value_and_grad(flat):
        _assign(np.asarray(flat, dtype=np.float64))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)
            loss = closure()
        grads = tape.gradient(loss, variables)
        grads = [tf.zeros_like(v) if g is None else g
                 for g, v in zip(grads, variables)]
        return (float(np.asarray(loss)),
                _pack([np.asarray(g) for g in grads]))

    x0 = _pack([np.asarray(v) for v in variables])
    result = _scipy_minimize(_value_and_grad, x0, jac=True, method=method,
                             options=options, callback=callback, **kwargs)
    _assign(np.asarray(result.x, dtype=np.float64))
    return result
