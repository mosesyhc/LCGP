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
    def __init__(self,
                 y: Optional[np.ndarray] = tf.Tensor,
                 x: Optional[np.ndarray] = tf.Tensor,
                 q: int = None,
                 var_threshold: float = None,
                 diag_error_structure: list = None,
                 parameter_clamp_flag: bool = False,
                 robust_mean: bool = True,
                 penalty_const: dict = None,
                 submethod: str = 'rep',
                 rep_standardize_ybar: bool = True,
                 verbose: bool = False):
        """
        Constructor for LCGP class.

        LCGP with optional replication support (set submethod='rep').
        """
        super().__init__()
        self.verbose = verbose
        self.robust_mean = robust_mean
        self.rep_standardize_ybar = rep_standardize_ybar            # can toggle this

        self.x = self._verify_data_types(x)
        self.y = self._verify_data_types(y)

        self.method = 'LCGP'
        self.submethod = 'rep'
        self.submethod_loss_map = {'rep':  self.neglpost_rep # replicated marginal likelihood
                                   }
        self.submethod_predict_map = {'rep':  self.predict_rep # replicated predictive dist.
                                      }

        self.parameter_clamp_flag = parameter_clamp_flag

        if (q is not None) and (var_threshold is not None):
            raise ValueError('Include only q or var_threshold but not both.')
        self.q = q
        self.var_threshold = var_threshold

        # Verify dims (raw inputs)
        self.n, self.d, self.p = self.verify_dim(self.y, self.x)

       # Keep raw copies for replication grouping
        self.x_orig = self.x
        self.y_orig = self.y

        # Standardize x/y    # SHARED
        self.x, self.x_min, self.x_max, _, self.xnorm = self.init_standard_x(self.x)

        # Replication
        self._rep_initialized = False

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

        self.Tks = None

        if diag_error_structure is None:
            self.diag_error_structure = [1] * int(self.p)
        else:
            self.diag_error_structure = diag_error_structure
        self.verify_error_structure(self.diag_error_structure, self.y)

        self.g, self.phi, self.diag_D, self.q = self.init_phi(var_threshold=var_threshold)

        # Initialize parameters
        self.lLmb = gpflow.Parameter(tf.ones([self.q, self.x.shape[1]], dtype=tf.float64),
                                     name='Latent GP log-scale',
                                     transform=tfp.bijectors.SoftClip(
                                         low=tf.constant(1e-6, dtype=tf.float64),
                                         high=tf.constant(1e4, dtype=tf.float64)
                                     ), dtype=tf.float64)
        self.lLmb0 = gpflow.Parameter(tf.ones([self.q], dtype=tf.float64),
                                      name='Latent GP log-lengthscale',
                                      transform=tfp.bijectors.SoftClip(
                                          low=tf.constant(1e-4, dtype=tf.float64),
                                          high=tf.constant(1e4, dtype=tf.float64)
                                      ), dtype=tf.float64)
        self.lsigma2s = gpflow.Parameter(tf.ones([len(self.diag_error_structure)], dtype=tf.float64),
                                         name='Diagonal error log-variance') #, transform=tfp.bijectors.Exp())
        self.lnugGPs = gpflow.Parameter(tf.ones([self.q], dtype=tf.float64) * 1e-6,
                                        name='Latent GP nugget scale',
                                        transform=tfp.bijectors.SoftClip(
                                            low=tf.math.exp(tf.constant(-16, dtype=tf.float64)),
                                            high=tf.math.exp(tf.constant(-2, dtype=tf.float64))
                                        ), dtype=tf.float64)

        if penalty_const is None:
            pc = {'lLmb': 40, 'lLmb0': 5}
        else:
            pc = penalty_const
            for k, v in pc.items():
                assert v >= 0, 'penalty constant should be nonnegative.'
        self.penalty_const = pc

        self.init_params()

        # placeholders for predictive quantities
        self.CinvMs = tf.fill([self.q, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Ths = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Th_hats = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Cinvhs = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.mks = tf.fill([self.q, self.n], tf.constant(float('nan'), dtype=tf.float64))

    # Replication preprocessing helpers
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

    # preprocess combined (not used)
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
            # (LaTeX: lines 540-546, 541-546)
            self.preprocess()  # builds x_unique, ybar, r, R, ybar_s, etc.
            # rebuild basis on ybar_s
            self._rep_initialized = True

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

    def __repr__(self):
        params = gpflow.utilities.tabulate_module_summary(self)
        desc = 'LCGP(\n' \
               '\tsubmethod:\t{:s}\n' \
               '\toutput dimension:\t{:d}\n' \
               '\tnumber of latent components:\t{:d}\n' \
               '\tparameter_clamping:\t{:s}\n' \
               '\trobust_standardization:\t{:s}\n' \
               '\tdiagonal_error structure:\t{:s}\n' \
               '\tparameters:\t\n{}\n)'.format(self.submethod, self.p,
                                         self.q, str(self.parameter_clamp_flag),
                                         str(self.robust_mean),
                                         str(self.diag_error_structure),
                                         params)
        return desc

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
            cumvar = np.cumsum(s**2) / np.sum(s**2)
            idx = np.argmax(cumvar > var_threshold)
            q = int(idx + 1) if np.any(cumvar > var_threshold) else p
        else:
            q = int(self.q)

        assert left_u.shape[1] == min(n, p)

        sing_q = singvals[:q]
        phi = left_u[:, :q] * tf.sqrt(tf.cast(n, tf.float64)) / sing_q
        # (LaTeX: D = diag(d_k), 549-550)
        diag_D = tf.reduce_sum(phi ** 2, axis=0)
        # (LaTeX around lines 256-290)
        g = tf.matmul(phi, y, transpose_a=True)
        print("======= VARIANCE OF G ======")
        print(tf.math.reduce_variance(g, axis=1, keepdims=False, name=None))

        return g, phi, diag_D, q

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

    def verify_dim(self, y, x):
        """
        Verifies if input and output dimensions match. Sets class variables for
        dimensions. Throws error if the dimensions do not match.
        """
        p, ny = tf.shape(y)[0], tf.shape(y)[1]
        nx, d = tf.shape(x)[0], tf.shape(x)[1]

        '''
        For raw replicates, likely have N rows in x and N columns in y (ny == nx == N).
        - Verify with (ybar_s, x_unique) so that
            ny == n_unique and nx == n_unique.
        '''

        assert ny == nx, 'Number of inputs (x) differs from number of outputs (y), y.shape[1] != x.shape[0]'

        return tf.constant(nx, tf.int32), tf.constant(d, tf.int32), tf.constant(p, tf.int32)

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

    def fit(self, verbose=False):  # SHARED
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

    def neglpost_rep(self):
        '''
        1) Build covariances on unique inputs:
              xk = self.x_unique       shape (n, d)
              Ck = K(xk, xk; θ_k)      shape (n, n)
        2) Build Σ^{-1/2}:
              If diagonal-by-block: use built_lsigma2s from get_param() to form per-dim scalings.
              If full Σ: use stored Cholesky/inverse-root.
        3) Basis from ybar:
              Φ shape (p, q), D = diag(d_k)
        4) b_k vector shape (n, 1):
              b_k = R @ Ybar @ (Σ^{-1/2} φ_k)
        5) Posterior precision and mean:
              s_k = C_k^{-1} + d_k R,   S_k = s_k^{-1},   m_k = S_k b_k
        6) Replicated negative log-marginal (up to constant):
            0.5 * sum_i r_i * ybar_i^T Σ^{-1} ybar_i
          - 0.5 * sum_k b_k^T S_k b_k
          + 0.5 * sum_k log|I + d_k R^{1/2} C_k R^{1/2}|
          + (n/2) log|Σ| - (p/2) log|R|
          + regularization
        '''
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        xk   = self.x_unique_s
        ybar = self.ybar_s
        r    = tf.cast(self.r, tf.float64)                          # r_i (LaTeX: lines 541-546)
        R    = self.R                                               # R = diag(r_i) (LaTeX: lines 541-546)
        n    = tf.cast(self.n, tf.float64)
        p    = tf.cast(self.p, tf.float64)

        D    = self.diag_D                                          # D = diag(d_k) (LaTeX: lines 549-550)
        phi  = self.phi                                             # Φ (LaTeX: lines 548-550)

        # Choose whether to standardize ybar inside the replicated likelihood
        use_std = getattr(self, "rep_standardize_ybar", True)

        sigma_var_raw = tf.exp(lsigma2s)                            # Σ (LaTeX: lines 566-569)
        sigma_inv_raw = 1.0 / sigma_var_raw                         # Σ^{-1}
        sigma_inv_sqrt_raw = tf.sqrt(sigma_inv_raw)                 # Σ^{-1/2}  (LaTeX: lines 589-591, 594-596)

        if use_std:
            # Standardized replicate-averages
            ybar = self.ybar_s

            # When ybar_s = (ybar_raw - mean)/std, the implied noise variance in standardized space is:
            #   Σ_std = Σ_raw / std^2
            std = self.ybar_std[:, 0]
            sigma_var_used = sigma_var_raw / tf.square(std)         # Σ_std
            sigma_inv_sqrt = sigma_inv_sqrt_raw * std               # Σ_std^{-1/2} = (1/sqrt(Σ_raw)) * std
        else:
            # Raw replicate-averages
            ybar = self.ybar
            sigma_var_used = sigma_var_raw                          # Σ_raw
            sigma_inv_sqrt = sigma_inv_sqrt_raw                     # Σ_raw^{-1/2}

        nlp = tf.constant(0.0, tf.float64)

        # 0.5 * sum_i r_i * ybar_i^T Σ^{-1} ybar_i                  (LaTeX: lines 693-694)
        ybar_scaled = ybar * sigma_inv_sqrt[:, None]            # Σ^{-1/2} \bar{Y} (LaTeX: 581-583)
        col_sq      = tf.reduce_sum(tf.square(ybar_scaled), axis=0)
        nlp += 0.5 * tf.reduce_sum(r * col_sq)                      # (LaTeX: 693-694)

        # + (n/2) log|Σ_s| = (n/2) (sum log Σ - 2 sum log std), (corresponds to the |Σ|^{-n/2} constant)
        nlp += 0.5 * n * tf.reduce_sum(tf.math.log(sigma_var_used))  #  (LaTeX: lines 690-691)

        # - (p/2) log|R| = - (p/2) * sum_i log(r_i)
        nlp += -0.5 * p * tf.reduce_sum(tf.math.log(r))             # (LaTeX: |R|^{p/2} in likelihood constant; 690-691)

        pc = self.penalty_const
        reg = (pc['lLmb'] * tf.reduce_sum(tf.square(tf.math.log(self.lLmb))) +
            pc['lLmb0'] * (2.0 / n) * tf.reduce_sum(tf.square(self.lLmb0.unconstrained_variable))
            - tf.reduce_sum(tf.math.log(tf.math.log(self.lnugGPs) + 100.0)))

        bkSb_sum = tf.constant(0.0, tf.float64)
        logA_sum = tf.constant(0.0, tf.float64)

        sr  = tf.sqrt(r)                                            # R^{1/2} via diag(sqrt(r_i)) (LaTeX: lines 706-709)

        q_int = tf.cast(self.q, tf.int32)
        for k in range(q_int):
            # C_k (LaTeX: lines 615-616, 712-713)
            Ck = Matern32(xk, xk, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

            # b_k = R @ ybar_s^T @ (Σ_s^{-1/2} φ_k)
            v_k = sigma_inv_sqrt * phi[:, k]                    # Σ^{-1/2} φ_k (LaTeX: 594-596)
            ytv = tf.linalg.matvec(tf.transpose(ybar), v_k)         # \bar{Y}^T (Σ^{-1/2} φ_k) (LaTeX: 594-596)
            b_k = r * ytv                                           # R @ (...) since R=diag(r) (LaTeX: 594-596, 625-628)

            d_k = D[k]

            # Below is a Woodbury-style way to compute S_k, b_k, etc. (LaTeX: lines 698-709)
            Cb  = tf.linalg.matvec(Ck, b_k)

            # A = I + d_k R^{1/2} C_k R^{1/2} (LaTeX: lines 703-704, 707-709)
            A   = tf.eye(self.n, dtype=tf.float64) + d_k * ((Ck * sr[None, :]) * sr[:, None])   # (LaTeX: lines 698-709)
            LA  = tf.linalg.cholesky(A)
            u   = tf.sqrt(d_k) * (sr * Cb)
            z   = tf.linalg.cholesky_solve(LA, tf.expand_dims(u, -1))
            z   = tf.squeeze(z, -1)
            Sb  = Cb - tf.linalg.matvec(Ck, (tf.sqrt(d_k) * (sr * z)))                          # (LaTeX: lines 692-695)

            bkSb_sum += tf.tensordot(b_k, Sb, axes=1)               # b_k^T S_k b_k term (LaTeX: +0.5 b_k^T V_k b_k, lines 693-695)
            logA_sum += 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LA)))               # log|A| where A = I + d_k R^{1/2} C_k R^{1/2} (LaTeX: lines 707-709)

        nlp += -0.5 * bkSb_sum                                      # (LaTeX: lines 693-695)
        nlp +=  0.5 * logA_sum

        nlp +=  reg
        nlp /= n
        return nlp

    def predict(self, x0, return_fullcov=False):  # SHARED
        """
        Returns predictive quantities at new input `x0`.  Both outputs are of
        size (number of new input, output dimension).
        :param x0: New input of size (number of new input, dimension of input).
        :param return_fullcov: Returns (predictive mean, predictive variance,
        variance for the true mean, full predictive covariance) if True.  Otherwise,
        only return the first three quantities.
        """
        x0 = self._verify_data_types(x0)
        submethod = self.submethod
        predict_map = self.submethod_predict_map
        try:
            predict_call = predict_map[submethod]
        except KeyError as e:
            print(e)
            # print('Invalid submethod.  Choices are \'full\', \'elbo\', or \'proflik\'.')
            raise KeyError('Invalid submethod.  Choices are \'full\'.')
        return tf.stop_gradient(predict_call(x0=x0, return_fullcov=return_fullcov))

    def _compute_aux_predictive_quantities_rep(self):
            lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
            xk   = self.x_unique_s
            r    = tf.cast(self.r, tf.float64)
            R    = self.R

            D    = self.diag_D
            phi  = self.phi  # (p, q)

            # standardized-ybar space or raw-ybar 
            use_std = getattr(self, "rep_standardize_ybar", True)
            if use_std:
                ybar = self.ybar_s  # (p, n)
            else:
                ybar = self.ybar    # (p, n)

            # Σ^{-1/2} on RAW scale
            sigma_inv_sqrt_raw = tf.exp(-0.5 * lsigma2s)  # (p,)

            # if using standardized ybar: ybar_s = (ybar_raw - mean)/std
            if use_std:
                std = self.ybar_std[:, 0]                 # (p,)
                sigma_inv_sqrt_used = sigma_inv_sqrt_raw * std
            else:
                sigma_inv_sqrt_used = sigma_inv_sqrt_raw

            self.psi_c = tf.transpose(phi) / sigma_inv_sqrt_used[:, None]  # (q,p), corresponds to Φ^T Σ^{-1/2}

            q = tf.cast(self.q, tf.int32)
            n = tf.cast(self.n, tf.int32)

            CinvM = tf.zeros([q, n], dtype=tf.float64)
            Tks   = tf.zeros([q, n, n], dtype=tf.float64)
            mks   = tf.zeros([q, n], dtype=tf.float64)

            sr  = tf.sqrt(r)

            for k in range(q):
                Ck = Matern32(xk, xk, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

                # b_k = R @ ybar^T @ (Σ^{-1/2} φ_k)
                v_k = sigma_inv_sqrt_used * phi[:, k]             # (p,)
                ytv = tf.linalg.matvec(tf.transpose(ybar), v_k)   # (n,)
                b_k = r * ytv                                     # (n,)

                d_k = D[k]

                # m_k = V_k b_k via Woodbury
                Cb  = tf.linalg.matvec(Ck, b_k)
                A   = tf.eye(n, dtype=tf.float64) + d_k * ((Ck * sr[None, :]) * sr[:, None])
                LA  = tf.linalg.cholesky(A)
                u   = tf.sqrt(d_k) * (sr * Cb)
                z   = tf.linalg.cholesky_solve(LA, tf.expand_dims(u, -1))
                z   = tf.squeeze(z, -1)
                m_k = Cb - tf.linalg.matvec(Ck, (tf.sqrt(d_k) * (sr * z)))

                # C_k^{-1} m_k = b_k - d_k R m_k
                CinvM_k = b_k - d_k * tf.linalg.matvec(R, m_k)

                # Build C_k^{-1} explicitly for T_k
                LC   = tf.linalg.cholesky(Ck)
                Id   = tf.eye(n, dtype=tf.float64)
                invC = tf.linalg.cholesky_solve(LC, Id)

                # P_k = C_k^{-1} + d_k R
                P_k = invC + d_k * R

                # V_k = P_k^{-1}
                V_k = tf.linalg.inv(P_k)

                # T_k = C_k^{-1} - C_k^{-1} V_k C_k^{-1}
                Tk  = invC - invC @ V_k @ invC

                CinvM = tf.tensor_scatter_nd_update(CinvM, [[k]], [CinvM_k])
                Tks   = tf.tensor_scatter_nd_update(Tks,   [[k]], [Tk])
                mks   = tf.tensor_scatter_nd_update(mks,   [[k]], [m_k])

            self.mks    = mks
            self.CinvMs = CinvM
            self.Tks    = Tks
            self.Ths    = None

    def predict_rep(self, x0, return_fullcov=False):
        need_aux = (self.Tks is None) or tf.reduce_any(tf.math.is_nan(self.CinvMs))
        if need_aux:
            self._compute_aux_predictive_quantities_rep()

        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        phi  = self.phi  # (p, q)

        Xtrain = self.x_unique_s
        Tks    = self.Tks
        CinvM  = self.CinvMs

        # standardize x0 into [0,1] using training mins/maxs
        x0 = self._verify_data_types(x0)
        x0 = (x0 - self.x_min) / (self.x_max - self.x_min)
        n0 = tf.shape(x0)[0]

        ghat = tf.zeros([self.q, n0], dtype=tf.float64)
        gvar = tf.zeros([self.q, n0], dtype=tf.float64)

        for k in range(self.q):
            c00k = Matern32(x0, x0,     llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k], diag_only=True)      # (n0,)
            c0k  = Matern32(x0, Xtrain, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k], diag_only=False)     # (n0,n)

            # μ_k(x_*) = c_k(x_*)^T (C_k^{-1} m_k) = c_k^T CinvM_k
            ghat_k = tf.linalg.matvec(c0k, CinvM[k])                                                        # (n0,)

            # σ_k^2(x_*) = c_{**} - c_k^T T_k c_k
            Tk     = Tks[k]                                                                                 # (n,n)
            v      = tf.matmul(c0k, Tk)                                                                     # (n0,n)
            quad   = tf.reduce_sum(v * c0k, axis=1)                                                         # (n0,)
            gvar_k = c00k - quad                                                                            # (n0,)

            ghat = tf.tensor_scatter_nd_update(ghat, [[k]], [ghat_k])
            gvar = tf.tensor_scatter_nd_update(gvar, [[k]], [gvar_k])

        self.ghat = ghat
        self.gvar = gvar

        # choose whether we are predicting in standardized ybar space or raw ybar space
        use_std = getattr(self, "rep_standardize_ybar", True)

        sigma_var_raw  = tf.exp(lsigma2s)              # (p,)
        sigma_sqrt_raw = tf.sqrt(sigma_var_raw)        # (p,)

        if use_std:
            std = self.ybar_std[:, 0]                  # (p,)
            # In standardized space: Σ_std = Σ_raw / std^2  =>  Σ_std^{1/2} = Σ_raw^{1/2} / std
            sigma_sqrt_used = sigma_sqrt_raw / std
            sigma_var_used  = sigma_var_raw / tf.square(std)
        else:
            sigma_sqrt_used = sigma_sqrt_raw
            sigma_var_used  = sigma_var_raw

        Psi = phi * sigma_sqrt_used[:, None]           # (p,q)

        predmean_used = tf.matmul(Psi, ghat)           # (p,n0)
        confvar_used  = tf.matmul(tf.square(Psi), gvar)  # (p,n0)
        predvar_used  = confvar_used + sigma_var_used[:, None]

        # if we predicted using standardized-ybar, unstandardize back to raw outputs
        if use_std:
            ypred    = predmean_used * self.ybar_std + self.ybar_mean
            yconfvar = confvar_used  * tf.square(self.ybar_std)
            ypredvar = predvar_used  * tf.square(self.ybar_std)
        else:
            ypred, yconfvar, ypredvar = predmean_used, confvar_used, predvar_used

        if return_fullcov:
            return ypred, ypredvar, yconfvar, None
        return ypred, ypredvar, yconfvar

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

    @staticmethod
    def verify_error_structure(diag_error_structure, y):
        """
        Verifies if diagonal error structure input, if any, is valid.
        """
        assert sum(diag_error_structure) == y.shape[0], \
            'Sum of error_structure should' \
            ' equal the output dimension.'

    def get_param(self):
        """
        Returns the parameters for LCGP instance.
        """
        lLmb, lLmb0, lsigma2s, lnugGPs = \
            self.lLmb, self.lLmb0, self.lsigma2s, self.lnugGPs

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
