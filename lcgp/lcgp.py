from ._import_util import _import_tensorflow
import tensorflow_probability as tfp
import gpflow
from .covmat import Matern32
import numpy as np

# for Python 3.9 inclusion
from typing import Optional

tf = _import_tensorflow()

# Display only code-breaking errors
tf.get_logger().setLevel('ERROR')
# Set default float type to float64
tf.keras.backend.set_floatx('float64')


class LCGP(gpflow.Module):
    """
    Latent Component Gaussian Process (LCGP)

    Supports two training/prediction paths:
      - submethod='full': uses all observations (x, y)
      - submethod='rep' : groups replicated x rows, uses (x_unique, ybar) structures
    """

    # =========================================================================
    # Constructor
    # =========================================================================
    def __init__(self,
                 y: Optional[np.ndarray] = tf.Tensor,
                 x: Optional[np.ndarray] = tf.Tensor,
                 q: int = None,
                 var_threshold: float = None,
                 diag_error_structure: list = None,
                 parameter_clamp_flag: bool = False,
                 robust_mean: bool = True,
                 penalty_const: dict = None,
                 submethod: str = 'full',
                 rep_standardize_ybar: bool = True,
                 verbose: bool = False):
        """
        Constructor for LCGP class.

        LCGP with optional replication support (set submethod='rep').
        """
        super().__init__()

        # -----------------------------
        # User toggles / config
        # -----------------------------
        self.verbose = verbose
        self.robust_mean = robust_mean
        self.rep_standardize_ybar = rep_standardize_ybar  # can toggle this
        self.parameter_clamp_flag = parameter_clamp_flag

        # -----------------------------
        # Verify input tensors
        # -----------------------------
        self.x = self._verify_data_types(x)
        self.y = self._verify_data_types(y)

        # -----------------------------
        # Mode selection (full vs rep)
        # -----------------------------
        self.method = 'LCGP'
        self.submethod = submethod
        self.submethod_loss_map = {'full': self.neglpost,
                                    'rep':  self.neglpost_rep # replicated marginal likelihood
                                    }
        self.submethod_predict_map = {'full': self.predict_full,
                                      'rep':  self.predict_rep # replicated predictive dist.
                                      }

        # -----------------------------
        # Latent dimension selection
        # -----------------------------
        if (q is not None) and (var_threshold is not None):
            raise ValueError('Include only q or var_threshold but not both.')
        self.q = q
        self.var_threshold = var_threshold

        # -----------------------------
        # Verify dims (raw inputs)
        # -----------------------------
        self.n, self.d, self.p = self.verify_dim(self.y, self.x)

        # Keep raw copies for replication grouping
        self.x_orig = self.x
        self.y_orig = self.y

        # -----------------------------
        # Standardize x (always)
        # -----------------------------
        self.x, self.x_min, self.x_max, _, self.xnorm = self.init_standard_x(self.x)

        # Replication
        self._rep_initialized = False

        # =====================================================================
        # Path A: Replicated preprocessing (submethod == 'rep')
        # =====================================================================
        if self.submethod == 'rep':
            # 1) resolve raw xy numpy
            xr, yr, N, d, p = self._get_raw_xy(x_raw=self.x_orig, y_raw=self.y_orig)

            # 2) group identical rows
            x_unique_np, inverse_np, counts_np = self._group_unique_rows_np(xr)
            n_unique = int(x_unique_np.shape[0])
            r_np = counts_np.astype(np.int32)

            # 3) compute replicate-averaged ybar
            ybar_np = self._compute_ybar_np(yr, inverse_np, n_unique)

            # 4) pack into TF tensors + x_unique_s + R
            (x_unique_tf,
             x_unique_s,
             group_ids_tf,
             r_tf,
             R_tf,
             ybar_tf) = self._pack_replication_tensors(
                x_unique_np=x_unique_np,
                inverse_np=inverse_np,
                r_np=r_np,
                ybar_np=ybar_np
            )

            # 5) compute standardization stats for ybar and standardized ybar_s
            ybar_mean_tf, ybar_std_tf = self._compute_center_spread_tf(ybar_tf)
            ybar_s_tf = (ybar_tf - ybar_mean_tf) / ybar_std_tf

            # 6) assign to self
            self.x_unique = x_unique_tf
            self.x_unique_s = x_unique_s
            self.group_ids = group_ids_tf
            self.r = r_tf
            self.R = R_tf
            self.ybar = ybar_tf
            self.ybar_s = ybar_s_tf
            self.ybar_mean = ybar_mean_tf
            self.ybar_std = ybar_std_tf

            # 7) reset (n,d,p) to unique counts
            self.n = tf.constant(n_unique, dtype=tf.int32)
            self.d = tf.constant(d, dtype=tf.int32)
            self.p = tf.constant(p, dtype=tf.int32)

            self._rep_initialized = True

        # =====================================================================
        # Path B: Full-data standardization (submethod == 'full')
        # =====================================================================
        elif self.submethod == 'full':
            self.y, self.ymean, self.ystd, _ = self.init_standard_y(self.y)

        else:
            raise ValueError('submethod should be full or rep.')

        # -----------------------------
        # Initialize basis (phi) and derived quantities
        # -----------------------------
        self.g, self.phi, self.diag_D, self.q = self.init_phi(var_threshold=var_threshold)

        self.Tks = None

        # -----------------------------
        # Error structure
        # -----------------------------
        if diag_error_structure is None:
            self.diag_error_structure = [1] * int(self.p)
        else:
            self.diag_error_structure = diag_error_structure

        self.verify_error_structure(self.diag_error_structure, self.y)

        # -----------------------------
        # Initialize parameters (GP + noise)
        # -----------------------------
        self.lLmb = gpflow.Parameter(
            tf.ones([self.q, self.x.shape[1]], dtype=tf.float64),
            name='Latent GP log-scale',
            transform=tfp.bijectors.SoftClip(
                low=tf.constant(1e-6, dtype=tf.float64),
                high=tf.constant(1e4, dtype=tf.float64)
            ),
            dtype=tf.float64
        )
        self.lLmb0 = gpflow.Parameter(
            tf.ones([self.q], dtype=tf.float64),
            name='Latent GP log-lengthscale',
            transform=tfp.bijectors.SoftClip(
                low=tf.constant(1e-4, dtype=tf.float64),
                high=tf.constant(1e4, dtype=tf.float64)
            ),
            dtype=tf.float64
        )
        self.lsigma2s = gpflow.Parameter(
            tf.ones([len(self.diag_error_structure)], dtype=tf.float64),
            name='Diagonal error log-variance'
        )
        self.lnugGPs = gpflow.Parameter(
            tf.ones([self.q], dtype=tf.float64) * 1e-6,
            name='Latent GP nugget scale',
            transform=tfp.bijectors.SoftClip(
                low=tf.math.exp(tf.constant(-16, dtype=tf.float64)),
                high=tf.math.exp(tf.constant(-2, dtype=tf.float64))
            ),
            dtype=tf.float64
        )

        # -----------------------------
        # Penalty constants / regularization
        # -----------------------------
        if penalty_const is None:
            pc = {'lLmb': 40, 'lLmb0': 5}
        else:
            pc = penalty_const
            for k, v in pc.items():
                assert v >= 0, 'penalty constant should be nonnegative.'
        self.penalty_const = pc

        self.init_params()

        # -----------------------------
        # Placeholders for predictive quantities
        # -----------------------------
        self.CinvMs = tf.fill([self.q, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Ths = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Th_hats = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Cinvhs = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.mks = tf.fill([self.q, self.n], tf.constant(float('nan'), dtype=tf.float64))

    # =========================================================================
    # Display
    # =========================================================================
    def __repr__(self):
        params = gpflow.utilities.tabulate_module_summary(self)
        desc = 'LCGP(\n' \
               '\tsubmethod:\t{:s}\n' \
               '\toutput dimension:\t{:d}\n' \
               '\tnumber of latent components:\t{:d}\n' \
               '\tparameter_clamping:\t{:s}\n' \
               '\trobust_standardization:\t{:s}\n' \
               '\tdiagonal_error structure:\t{:s}\n' \
               '\tparameters:\t\n{}\n)'.format(
                    self.submethod, self.p,
                    self.q, str(self.parameter_clamp_flag),
                    str(self.robust_mean),
                    str(self.diag_error_structure),
                    params
               )
        return desc

    # =========================================================================
    # Utils: type checks, dims, transforms
    # =========================================================================
    @staticmethod
    def _verify_data_types(t):
        """
        Verify if inputs are TensorFlow tensors, if not, cast into tensors.
        Verify if inputs are at least 2-dimensional, if not, expand dimensions to 2.
        """
        if not isinstance(t, tf.Tensor):
            t = tf.convert_to_tensor(t, dtype=tf.float64)
        if t.ndim < 2:
            t = tf.expand_dims(t, axis=1)
        return t

    def verify_dim(self, y, x):
        """
        Verifies if input and output dimensions match. Sets class variables for
        dimensions. Throws error if the dimensions do not match.
        """
        p, ny = tf.shape(y)[0], tf.shape(y)[1]
        nx, d = tf.shape(x)[0], tf.shape(x)[1]

        assert ny == nx, 'Number of inputs (x) differs from number of outputs (y), y.shape[1] != x.shape[0]'

        return tf.constant(nx, tf.int32), tf.constant(d, tf.int32), tf.constant(p, tf.int32)

    @staticmethod
    def verify_error_structure(diag_error_structure, y):
        """
        Verifies if diagonal error structure input, if any, is valid.
        """
        assert sum(diag_error_structure) == y.shape[0], \
            'Sum of error_structure should equal the output dimension.'

    def tx_x(self, xs):
        """
        Reverts standardization of inputs.
        """
        return xs * (self.x_max - self.x_min) + self.x_min

    def tx_y(self, ys):
        """
        Reverts output standardization.
        """
        return ys * self.ystd + self.ymean

    # =========================================================================
    # Standardization 
    # =========================================================================
    @staticmethod
    def init_standard_x(x):
        """
        Standardizes training inputs and collects summary information.
        """
        x_max = tf.reduce_max(x, axis=0)
        x_min = tf.reduce_min(x, axis=0)
        xs = (x - x_min) / (x_max - x_min)

        xnorm = tf.zeros(x.shape[1], dtype=tf.float64)
        for j in range(x.shape[1]):
            xdist = tf.abs((tf.reshape(x[:, j], (-1, 1)) - x[:, j]))
            positive_xdist = tf.boolean_mask(xdist, xdist > 0)
            mean_val = tf.reduce_mean(positive_xdist)
            xnorm = tf.tensor_scatter_nd_update(xnorm, [[j]], [mean_val])
        return xs, x_min, x_max, x, xnorm

    def init_standard_y(self, y):
        """
        Standardizes outputs and collects summary information.
        """
        if self.robust_mean:
            ycenter = tfp.stats.percentile(y, 50.0, axis=1, keepdims=True)
            yspread = tfp.stats.percentile(tf.abs(y - ycenter), 50.0, axis=1, keepdims=True)
        else:
            ycenter = tf.reduce_mean(y, axis=1, keepdims=True)
            yspread = tf.math.reduce_std(y, axis=1, keepdims=True)

        ys = (y - ycenter) / yspread
        return ys, ycenter, yspread, y

    # =========================================================================
    # Replication preprocessing helpers
    # =========================================================================
    def _get_raw_xy(self, x_raw=None, y_raw=None):
        """
        Resolve raw-scale x/y
        """
        if x_raw is None:
            x_raw = self.x_orig
        if y_raw is None:
            y_raw = self.y_orig

        xr = x_raw.numpy() if isinstance(x_raw, tf.Tensor) else np.asarray(x_raw)
        yr = y_raw.numpy() if isinstance(y_raw, tf.Tensor) else np.asarray(y_raw)

        assert xr.ndim == 2, "x_raw must be (N, d)"
        assert yr.ndim == 2, "y_raw must be (p, N)"
        N, d = xr.shape
        p, Ny = yr.shape
        assert Ny == N, "y_raw columns must match x_raw rows"

        return xr, yr, N, d, p

    def _group_unique_rows_np(self, xr):
        """
        Group identical rows of xr
        """
        x_unique, inverse, counts = np.unique(
            xr, axis=0, return_inverse=True, return_counts=True
        )
        return x_unique, inverse, counts

    def _compute_ybar_np(self, yr, inverse, n):
        """
        Compute replicate-averaged outputs ybar on RAW scale
        """
        p, N = yr.shape
        ybar = np.zeros((p, n), dtype=np.float64)
        for i in range(n):
            cols = (inverse == i)
            ybar[:, i] = yr[:, cols].mean(axis=1)
        return ybar

    def _pack_replication_tensors(self, x_unique_np, inverse_np, r_np, ybar_np):
        """
        Convert numpy replication structures
        """
        x_unique_tf = tf.convert_to_tensor(x_unique_np, dtype=tf.float64)     # (n,d)
        x_unique_s = (x_unique_tf - self.x_min) / (self.x_max - self.x_min)   # (n,d)

        group_ids_tf = tf.convert_to_tensor(inverse_np, dtype=tf.int32)       # (N,)
        r_tf = tf.convert_to_tensor(r_np, dtype=tf.int32)                     # (n,)
        R_tf = tf.linalg.diag(tf.cast(r_tf, tf.float64))                      # (n,n)
        ybar_tf = tf.convert_to_tensor(ybar_np, dtype=tf.float64)             # (p,n)

        return x_unique_tf, x_unique_s, group_ids_tf, r_tf, R_tf, ybar_tf

    def _compute_center_spread_tf(self, Y):
        """
        Compute (center, spread) per output dim for standardization
        """
        if self.robust_mean:
            ycenter = tfp.stats.percentile(Y, 50.0, axis=1, keepdims=True)
            yspread = tfp.stats.percentile(tf.abs(Y - ycenter), 50.0, axis=1, keepdims=True)
        else:
            ycenter = tf.reduce_mean(Y, axis=1, keepdims=True)
            yspread = tf.math.reduce_std(Y, axis=1, keepdims=True)

        yspread = tf.where(yspread > 0, yspread, tf.ones_like(yspread, dtype=tf.float64))
        return ycenter, yspread

    def preprocess(self, y_raw=None, x_raw=None):
        """
        Returns a tuple of replication structures
        """
        xr, yr, N, d, p = self._get_raw_xy(x_raw=x_raw, y_raw=y_raw)
        x_unique_np, inverse_np, counts_np = self._group_unique_rows_np(xr)
        n_unique = int(x_unique_np.shape[0])
        r_np = counts_np.astype(np.int32)
        ybar_np = self._compute_ybar_np(yr, inverse_np, n_unique)

        (x_unique_tf,
         x_unique_s,
         group_ids_tf,
         r_tf,
         R_tf,
         ybar_tf) = self._pack_replication_tensors(
            x_unique_np=x_unique_np,
            inverse_np=inverse_np,
            r_np=r_np,
            ybar_np=ybar_np
        )

        ybar_mean_tf, ybar_std_tf = self._compute_center_spread_tf(ybar_tf)
        ybar_s_tf = (ybar_tf - ybar_mean_tf) / ybar_std_tf

        return (
            x_unique_tf, x_unique_s, group_ids_tf, r_tf, R_tf,
            ybar_tf, ybar_s_tf, ybar_mean_tf, ybar_std_tf,
            tf.constant(n_unique, tf.int32), tf.constant(d, tf.int32), tf.constant(p, tf.int32)
        )

    def _ensure_replication(self):
        """
        Build replication structures once if not yet built.
        """
        if not self._rep_initialized:
            self.preprocess()
            self._rep_initialized = True

    # =========================================================================
    # Phi / basis init (Replication)
    # =========================================================================
    def _get_phi_input(self):
        """
        Choose which Y matrix to use for SVD basis.
        Replicated: ybar_s if rep_standardize_ybar True and available;
        Full: use y.
        """
        if self.submethod != "rep":
            return self.y

        if getattr(self, "rep_standardize_ybar", True) and hasattr(self, "ybar_s"):
            return self.ybar_s
        if hasattr(self, "ybar"):
            return self.ybar
        return self.y

    def init_phi(self, var_threshold: float = None):
        """
        Initialization of orthogonal basis, computed with SVD.
        Uses ybar_s if replication, else y.
        """
        y = self._get_phi_input()

        n = int(self.n.numpy())
        p = int(self.p.numpy())

        singvals, left_u, _ = tf.linalg.svd(y, full_matrices=False)

        if (self.q is None) and (var_threshold is None):
            q = p
        elif (self.q is None) and (var_threshold is not None):
            s = singvals.numpy()
            cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
            idx = np.argmax(cumvar > var_threshold)
            q = int(idx + 1) if np.any(cumvar > var_threshold) else p
        else:
            q = int(self.q)

        assert left_u.shape[1] == min(n, p)

        sing_q = singvals[:q]
        phi = left_u[:, :q] * tf.sqrt(tf.cast(n, tf.float64)) / sing_q
        diag_D = tf.reduce_sum(phi ** 2, axis=0)
        g = tf.matmul(phi, y, transpose_a=True)
        print("======= VARIANCE OF G ======")
        print(tf.math.reduce_variance(g, axis=1, keepdims=False, name=None))

        return g, phi, diag_D, q

    # =========================================================================
    # Parameters / initialization
    # =========================================================================
    def init_params(self):
        """
        Initializes parameters for LCGP.
        """
        x = self.x
        d = self.d

        llmb = np.exp(0.5 * np.log(d) + np.log(np.std(x, axis=0)))
        lLmb = np.tile(llmb, self.q).reshape((self.q, d))
        lLmb0 = np.ones(self.q, dtype=np.float64)
        lnugGPs = np.exp(-10.) * np.ones(self.q, dtype=np.float64)

        err_struct = self.diag_error_structure
        lsigma2_diag = np.zeros(len(err_struct), dtype=np.float64)
        col = 0
        for k in range(len(err_struct)):
            lsigma2_diag[k] = np.log(np.var(self.y[col:(col + err_struct[k])]))
            col += err_struct[k]

        self.lLmb.assign(lLmb)
        self.lLmb0.assign(lLmb0)
        self.lnugGPs.assign(lnugGPs)
        self.lsigma2s.assign(lsigma2_diag)
        return

    def get_param(self):
        """
        Returns the parameters for LCGP instance.
        """
        lLmb, lLmb0, lsigma2s, lnugGPs = self.lLmb, self.lLmb0, self.lsigma2s, self.lnugGPs

        built_lsigma2s = tf.zeros(self.p, dtype=tf.float64)
        err_struct = self.diag_error_structure
        col = 0
        for k in range(len(err_struct)):
            built_lsigma2s = tf.tensor_scatter_nd_update(
                built_lsigma2s,
                tf.range(col, col + err_struct[k])[:, tf.newaxis],
                tf.fill([err_struct[k]], lsigma2s[k])
            )
            col += err_struct[k]

        return lLmb, lLmb0, built_lsigma2s, lnugGPs

    # =========================================================================
    # Training / loss dispatch
    # =========================================================================
    def fit(self, verbose=False):
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.loss, self.trainable_variables, compile=False)
        return

    def loss(self):
        """
        Computes the loss based on the submethod.
        """
        try:
            return self.submethod_loss_map[self.submethod]()
        except KeyError:
            raise ValueError("Invalid submethod. Choices are 'full' or 'rep'.")

    # =========================================================================
    # Loss: replicated
    # =========================================================================
    @tf.function
    def neglpost_rep(self):
        '''
        Replicated negative log marginal (up to constants), matching your working rep file.
        '''
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        xk = self.x_unique_s

        r = tf.cast(self.r, tf.float64)
        R = self.R
        n = tf.cast(self.n, tf.float64)
        p = tf.cast(self.p, tf.float64)

        D = self.diag_D
        phi = self.phi

        use_std = getattr(self, "rep_standardize_ybar", True)

        sigma_var_raw = tf.exp(lsigma2s)              # (p,)
        sigma_inv_raw = 1.0 / sigma_var_raw           # (p,)
        sigma_inv_sqrt_raw = tf.sqrt(sigma_inv_raw)   # (p,)

        if use_std:
            ybar = self.ybar_s
            std = self.ybar_std[:, 0]
            sigma_var_used = sigma_var_raw / tf.square(std)
            sigma_inv_sqrt = sigma_inv_sqrt_raw * std
        else:
            ybar = self.ybar
            sigma_var_used = sigma_var_raw
            sigma_inv_sqrt = sigma_inv_sqrt_raw

        nlp = tf.constant(0.0, tf.float64)

        # 0.5 * sum_i r_i * ybar_i^T Σ^{-1} ybar_i
        ybar_scaled = ybar * sigma_inv_sqrt[:, None]
        col_sq = tf.reduce_sum(tf.square(ybar_scaled), axis=0)
        nlp += 0.5 * tf.reduce_sum(r * col_sq)

        # + (n/2) log|Σ_used|
        nlp += 0.5 * n * tf.reduce_sum(tf.math.log(sigma_var_used))

        # - (p/2) log|R|
        nlp += -0.5 * p * tf.reduce_sum(tf.math.log(r))

        sr = tf.sqrt(r)

        bkSb_sum = tf.constant(0.0, tf.float64)
        logA_sum = tf.constant(0.0, tf.float64)

        q_int = tf.cast(self.q, tf.int32)
        for k in range(q_int):
            Ck = Matern32(xk, xk, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

            v_k = sigma_inv_sqrt * phi[:, k]
            ytv = tf.linalg.matvec(tf.transpose(ybar), v_k)
            b_k = r * ytv

            d_k = D[k]

            Cb = tf.linalg.matvec(Ck, b_k)

            A = tf.eye(self.n, dtype=tf.float64) + d_k * ((Ck * sr[None, :]) * sr[:, None])
            LA = tf.linalg.cholesky(A)
            u = tf.sqrt(d_k) * (sr * Cb)
            z = tf.linalg.cholesky_solve(LA, tf.expand_dims(u, -1))
            z = tf.squeeze(z, -1)
            Sb = Cb - tf.linalg.matvec(Ck, (tf.sqrt(d_k) * (sr * z)))

            bkSb_sum += tf.tensordot(b_k, Sb, axes=1)
            logA_sum += 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LA)))

        nlp += -0.5 * bkSb_sum
        nlp += 0.5 * logA_sum

        nlp /= n
        return nlp

    # =========================================================================
    # Loss: full
    # =========================================================================
    @tf.function
    def neglpost(self):
        # print('in neg log normal')
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        x = self.x
        y = self.y

        pc = self.penalty_const

        n = self.n
        q = self.q
        D = self.diag_D
        phi = self.phi
        psi_c = tf.transpose(phi) / tf.sqrt(tf.exp(lsigma2s))

        nlp = tf.constant(0., dtype=tf.float64)

        for k in range(q):
            Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])
            Wk, Uk = tf.linalg.eigh(Ck)

            Qk = tf.matmul(Uk, tf.matmul(tf.linalg.diag(1 / (D[k] + 1 / Wk)), tf.transpose(Uk)))
            Pk = tf.matmul(tf.expand_dims(psi_c[k], axis=1), tf.expand_dims(psi_c[k], axis=0))

            yQk = tf.matmul(y, Qk)
            yPk = tf.matmul(tf.transpose(y), tf.transpose(Pk))

            nlp += (0.5 * tf.reduce_sum(tf.math.log(1 + D[k] * Wk)))
            nlp += -(0.5 * tf.reduce_sum(yQk * tf.transpose(yPk)))

        nlp += (n / 2 * tf.reduce_sum(lsigma2s))
        nlp += (0.5 * tf.reduce_sum(tf.square(tf.transpose(y) / tf.sqrt(tf.exp(lsigma2s)))))

        # Regularization
        # nlp += (pc['lLmb'] * tf.reduce_sum(tf.square(tf.math.log(lLmb))) +
        #         pc['lLmb0'] * (2 / n) * tf.reduce_sum(tf.square(lLmb0.unconstrained_variable)))
        # nlp += (-tf.reduce_sum(tf.math.log(tf.math.log(lnugGPs) + 100)))
        # nlp /= tf.cast(n, tf.float64)
        return nlp

    # =========================================================================
    # Prediction dispatch
    # =========================================================================
    def predict(self, x0, return_fullcov=False):
        x0 = self._verify_data_types(x0)
        submethod = self.submethod
        predict_map = self.submethod_predict_map
        try:
            predict_call = predict_map[submethod]
        except KeyError as e:
            print(e)
            raise KeyError('Invalid submethod.  Choices are \'full\' or \'rep\'.')
        return tf.stop_gradient(predict_call(x0=x0, return_fullcov=return_fullcov))

    # =========================================================================
    # Aux predictive quantities 
    # =========================================================================
    def compute_aux_predictive_quantities(self):
        """
        Compute auxiliary quantities for predictions using full posterior approach.
        """
        if hasattr(self, 'x_unique') and hasattr(self, 'ybar'):
            self._compute_aux_predictive_quantities_rep()
            return

        x = self.x
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()

        D = self.diag_D
        B = tf.matmul(tf.transpose(self.y) / tf.sqrt(tf.exp(lsigma2s)), self.phi)

        CinvM = tf.zeros([self.q, self.n], dtype=tf.float64)
        Th = tf.zeros([self.q, self.n, self.n], dtype=tf.float64)

        for k in range(self.q):
            Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])
            Wk, Uk = tf.linalg.eigh(Ck)

            IpdkCkinv = tf.matmul(Uk, tf.matmul(tf.linalg.diag(1.0 / (1.0 + D[k] * Wk)), tf.transpose(Uk)))

            CkinvMk = tf.linalg.matvec(IpdkCkinv, tf.transpose(B)[k])
            Thk = tf.matmul(
                Uk,
                tf.matmul(tf.linalg.diag(tf.sqrt((D[k] * Wk ** 2) / (Wk ** 2 + D[k] * Wk ** 3))), tf.transpose(Uk))
            )

            CinvM = tf.tensor_scatter_nd_update(CinvM, [[k]], tf.expand_dims(CkinvMk, axis=0))
            Th = tf.tensor_scatter_nd_update(Th, [[k]], tf.expand_dims(Thk, axis=0))

        self.CinvMs = CinvM
        self.Ths = Th

    def _compute_aux_predictive_quantities_rep(self):
        """
        MATCH TOP FILE:
        - support rep_standardize_ybar toggle
        - build psi_c consistently with the space used
        - compute (CinvM, Tks, mks) the same way
        """
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        xk = self.x_unique_s
        r = tf.cast(self.r, tf.float64)
        R = self.R

        D = self.diag_D
        phi = self.phi  # (p,q)

        use_std = getattr(self, "rep_standardize_ybar", True)
        if use_std:
            ybar = self.ybar_s
        else:
            ybar = self.ybar

        sigma_inv_sqrt_raw = tf.exp(-0.5 * lsigma2s)  # (p,)
        if use_std:
            std = self.ybar_std[:, 0]
            sigma_inv_sqrt_used = sigma_inv_sqrt_raw * std
        else:
            sigma_inv_sqrt_used = sigma_inv_sqrt_raw

        # corresponds to Φ^T Σ^{-1/2}
        self.psi_c = tf.transpose(phi) / sigma_inv_sqrt_used[:, None]  # (q,p)

        q = tf.cast(self.q, tf.int32)
        n = tf.cast(self.n, tf.int32)

        CinvM = tf.zeros([q, n], dtype=tf.float64)
        Tks = tf.zeros([q, n, n], dtype=tf.float64)
        mks = tf.zeros([q, n], dtype=tf.float64)

        sr = tf.sqrt(r)

        for k in range(q):
            Ck = Matern32(xk, xk, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

            # b_k = R @ ybar^T @ (Σ^{-1/2} φ_k)
            v_k = sigma_inv_sqrt_used * phi[:, k]               # (p,)
            ytv = tf.linalg.matvec(tf.transpose(ybar), v_k)     # (n,)
            b_k = r * ytv                                       # (n,)

            d_k = D[k]

            # m_k = V_k b_k via Woodbury
            Cb = tf.linalg.matvec(Ck, b_k)
            A = tf.eye(n, dtype=tf.float64) + d_k * ((Ck * sr[None, :]) * sr[:, None])
            LA = tf.linalg.cholesky(A)
            u = tf.sqrt(d_k) * (sr * Cb)
            z = tf.linalg.cholesky_solve(LA, tf.expand_dims(u, -1))
            z = tf.squeeze(z, -1)
            m_k = Cb - tf.linalg.matvec(Ck, (tf.sqrt(d_k) * (sr * z)))

            # C_k^{-1} m_k = b_k - d_k R m_k
            CinvM_k = b_k - d_k * tf.linalg.matvec(R, m_k)

            # Build C_k^{-1} explicitly for T_k
            LC = tf.linalg.cholesky(Ck)
            Id = tf.eye(n, dtype=tf.float64)
            invC = tf.linalg.cholesky_solve(LC, Id)

            # P_k = C_k^{-1} + d_k R
            P_k = invC + d_k * R

            # V_k = P_k^{-1}
            V_k = tf.linalg.inv(P_k)

            # T_k = C_k^{-1} - C_k^{-1} V_k C_k^{-1}
            Tk = invC - invC @ V_k @ invC

            CinvM = tf.tensor_scatter_nd_update(CinvM, [[k]], [CinvM_k])
            Tks = tf.tensor_scatter_nd_update(Tks, [[k]], [Tk])
            mks = tf.tensor_scatter_nd_update(mks, [[k]], [m_k])

        self.mks = mks
        self.CinvMs = CinvM
        self.Tks = Tks
        self.Ths = None

    # =========================================================================
    # Prediction: full
    # =========================================================================
    def predict_full(self, x0, return_fullcov=False):
        """
        Returns predictions using full posterior approach.
        """
        if tf.reduce_any(tf.math.is_nan(self.CinvMs)) or tf.reduce_any(tf.math.is_nan(self.Ths)):
            self.compute_aux_predictive_quantities()

        x = self.x
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        phi = self.phi
        CinvM = self.CinvMs
        Th = self.Ths

        x0 = self._verify_data_types(x0)
        x0 = (x0 - self.x_min) / (self.x_max - self.x_min)
        n0 = tf.shape(x0)[0]

        ghat = tf.zeros([self.q, n0], dtype=tf.float64)
        gvar = tf.zeros([self.q, n0], dtype=tf.float64)
        for k in range(self.q):
            c00k = Matern32(x0, x0, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k], diag_only=True)
            c0k = Matern32(x0, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k], diag_only=False)

            ghat_k = tf.linalg.matvec(c0k, CinvM[k])
            gvar_k = c00k - tf.reduce_sum(tf.square(tf.matmul(c0k, Th[k])), axis=1)

            ghat = tf.tensor_scatter_nd_update(ghat, [[k]], [ghat_k])
            gvar = tf.tensor_scatter_nd_update(gvar, [[k]], [gvar_k])

        self.ghat = ghat
        self.gvar = gvar

        psi = tf.transpose(phi) * tf.sqrt(tf.exp(lsigma2s))

        predmean = tf.matmul(psi, ghat, transpose_a=True)
        confvar = tf.matmul(tf.transpose(gvar), tf.square(psi))
        predvar = confvar + tf.exp(lsigma2s)

        ypred = self.tx_y(predmean)
        yconfvar = tf.transpose(confvar) * tf.square(self.ystd)
        ypredvar = tf.transpose(predvar) * tf.square(self.ystd)

        if return_fullcov:
            CH = tf.sqrt(gvar)[..., tf.newaxis] * psi[tf.newaxis, ...]
            yfullpredcov = (tf.einsum('nij,njk->nik', CH, tf.transpose(CH, perm=[0, 2, 1])) +
                            tf.linalg.diag(tf.exp(lsigma2s)))
            yfullpredcov *= tf.square(self.ystd)
            return ypred, ypredvar, yconfvar, yfullpredcov

        return ypred, ypredvar, yconfvar

    # =========================================================================
    # Prediction: replicated
    # =========================================================================
    def predict_rep(self, x0, return_fullcov=False):
        need_aux = (self.Tks is None) or tf.reduce_any(tf.math.is_nan(self.CinvMs))
        if need_aux:
            self._compute_aux_predictive_quantities_rep()

        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        phi = self.phi  # (p,q)

        Xtrain = self.x_unique_s
        Tks = self.Tks
        CinvM = self.CinvMs

        x0 = self._verify_data_types(x0)
        x0 = (x0 - self.x_min) / (self.x_max - self.x_min)
        n0 = tf.shape(x0)[0]

        ghat = tf.zeros([self.q, n0], dtype=tf.float64)
        gvar = tf.zeros([self.q, n0], dtype=tf.float64)

        for k in range(self.q):
            c00k = Matern32(x0, x0, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k], diag_only=True)
            c0k = Matern32(x0, Xtrain, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k], diag_only=False)

            # mean
            ghat_k = tf.linalg.matvec(c0k, CinvM[k])

            # var
            Tk = Tks[k]
            v = tf.matmul(c0k, Tk)
            quad = tf.reduce_sum(v * c0k, axis=1)
            gvar_k = c00k - quad

            ghat = tf.tensor_scatter_nd_update(ghat, [[k]], [ghat_k])
            gvar = tf.tensor_scatter_nd_update(gvar, [[k]], [gvar_k])

        self.ghat = ghat
        self.gvar = gvar

        use_std = getattr(self, "rep_standardize_ybar", True)

        sigma_var_raw = tf.exp(lsigma2s)       # (p,)
        sigma_sqrt_raw = tf.sqrt(sigma_var_raw)

        if use_std:
            std = self.ybar_std[:, 0]         # (p,)
            sigma_sqrt_used = sigma_sqrt_raw / std
            sigma_var_used = sigma_var_raw / tf.square(std)
        else:
            sigma_sqrt_used = sigma_sqrt_raw
            sigma_var_used = sigma_var_raw

        Psi = phi * sigma_sqrt_used[:, None]   # (p,q)

        predmean_used = tf.matmul(Psi, ghat)               # (p,n0)
        confvar_used = tf.matmul(tf.square(Psi), gvar)     # (p,n0)
        predvar_used = confvar_used + sigma_var_used[:, None]

        if use_std:
            ypred = predmean_used * self.ybar_std + self.ybar_mean
            yconfvar = confvar_used * tf.square(self.ybar_std)
            ypredvar = predvar_used * tf.square(self.ybar_std)
        else:
            ypred, yconfvar, ypredvar = predmean_used, confvar_used, predvar_used

        if return_fullcov:
            return ypred, ypredvar, yconfvar, None
        return ypred, ypredvar, yconfvar