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
                 submethod: str = 'full',
                 verbose: bool = False):
        """
        Constructor for LCGP class.

        LCGP with optional replication support (set submethod='rep').
        """
        super().__init__()
        self.verbose = verbose
        self.robust_mean = robust_mean
        self.x = self._verify_data_types(x)
        self.y = self._verify_data_types(y)

        self.method = 'LCGP'
        self.submethod = submethod
        self.submethod_loss_map = {'full': self.neglpost,
                                   'rep':  self.neglpost_rep # replicated marginal likelihood
                                   }
        self.submethod_predict_map = {'full': self.predict_full,
                                      'rep':  self.predict_rep # replicated predictive dist.
                                      }

        self.parameter_clamp_flag = parameter_clamp_flag

        if (q is not None) and (var_threshold is not None):
            raise ValueError('Include only q or var_threshold but not both.')
        self.q = q
        self.var_threshold = var_threshold

        '''
        Precompute replicate groups before any standardizations.
        - Group identical input rows in self.x
        - Create:
            self.x_unique   : shape (n, d) unique inputs
            self.group_ids  : list of length N mapping each row in x to unique index i
            self.r          : shape (n, ) counts r_i (replicated per unique input)
            self.R          : shape (n, n) diag of r
        - Compute replicate-averaged outputs:
            self.ybar       : shape (p, n)
        '''

        # standardize x to unit hypercube
        self.x, self.x_min, self.x_max, self.x_orig, self.xnorm = \
            self.init_standard_x(self.x)
        
        '''
        Standardize y 
        - Standardize per output dim across replicates
        - For replicated likelihood, create standardized ybar:
            self.ybar, self.ybar_mean, self.ybar_std
                by averaging raw y in original scale, then standardize the averages
        '''

        # standardize y
        self.y, self.ymean, self.ystd, self.y_orig = self.init_standard_y(self.y)

        # placeholders for variables
        self.n, self.d, self.p = 0., 0., 0.
        # verify that input and output dimensions match
        # sets n, d, and p
        self.verify_dim(self.y, self.x)

        '''
        After verify_dim, reset (n, d) to unique counts if we use x_unique
            self.n = n_unique
            self.d = d
        '''

        # reset q if none is provided
        self.g, self.phi, self.diag_D, self.q = \
            self.init_phi(var_threshold=var_threshold)

        if diag_error_structure is None:
            self.diag_error_structure = [1] * int(self.p)
        else:
            self.diag_error_structure = diag_error_structure

        self.verify_error_structure(self.diag_error_structure, self.y)

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
        # C^{-1}mk, T^{1/2}, \hat{T}^{1/2}, C^{-1/2}
        self.CinvMs = tf.fill([self.q, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Ths = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Th_hats = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Cinvhs = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))

        self._rep_initialized = False

    def _ensure_replication(self):
        """
        Build replication structures once if not yet built.
        """
        if not self._rep_initialized:
            self.preprocess()  # builds x_unique, ybar, r, R, ybar_s, etc.
            # rebuild basis on ybar_s
            self.g, self.phi, self.diag_D, self.q = self.init_phi(var_threshold=self.var_threshold)
            self.CinvMs = tf.fill([self.q, self.n], tf.constant(float('nan'), dtype=tf.float64))
            self.Ths    = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
            self.Tks    = None
            self._rep_initialized = True

    def preprocess(self, y_raw=None, x_raw=None):
        # ADD SELF.IS_REP: shud i use replication structure??
        """
        Build replicate structure.
        Sets:
            self.x_unique : (n, d) unique inputs (raw scale)
            self.group_ids: (N,) int indices mapping each raw sample -> unique row
            self.r        : (n,) replicate counts per unique input (int32)
            self.R        : (n, n) diag(r) (float64)
            self.ybar     : (p, n) replicate-averaged outputs on RAW scale (float64)
            self.ybar_s   : (p, n) standardized ybar (float64)
            self.ybar_mean: (p, 1) center used to standardize ybar (float64)
            self.ybar_std : (p, 1) spread used to standardize ybar (float64)
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

        x_unique, inverse, counts = np.unique(xr, axis=0, return_inverse=True, return_counts=True)
        n = x_unique.shape[0]                  
        r = counts.astype(np.int32)             

        ybar = np.zeros((p, n), dtype=np.float64)
        for i in range(n):
            cols = (inverse == i)
            ybar[:, i] = yr[:, cols].mean(axis=1)

        self.x_unique  = tf.convert_to_tensor(x_unique, dtype=tf.float64)   # (n, d)
        self.group_ids = tf.convert_to_tensor(inverse,  dtype=tf.int32)     # (N,)
        self.r         = tf.convert_to_tensor(r,        dtype=tf.int32)     # (n,)
        self.R         = tf.linalg.diag(tf.cast(self.r, tf.float64))        # (n, n)
        self.ybar      = tf.convert_to_tensor(ybar,     dtype=tf.float64)   # (p, n)

        if self.robust_mean:
            ycenter = tfp.stats.percentile(self.ybar, 50.0, axis=1, keepdims=True)
            yspread = tfp.stats.percentile(tf.abs(self.ybar - ycenter), 50.0, axis=1, keepdims=True)
        else:
            ycenter = tf.reduce_mean(self.ybar, axis=1, keepdims=True)
            yspread = tf.math.reduce_std(self.ybar, axis=1, keepdims=True)

        yspread = tf.where(yspread > 0, yspread, tf.ones_like(yspread, dtype=tf.float64))
        self.ybar_s   = (self.ybar - ycenter) / yspread
        self.ybar_mean = ycenter
        self.ybar_std  = yspread

        self.n = tf.constant(n,  dtype=tf.int32)
        self.d = tf.constant(d,  dtype=tf.int32)
        self.p = tf.constant(p,  dtype=tf.int32)

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

    '''
    For replicated model, will likely want:
          self.init_standard_ybar(y_raw, group_ids) -> (ybar_s, ybar_mean, ybar_std, ybar_raw)
            where ybar_raw is built first by averaging replicates per unique input
            and ybar_s = (ybar_raw - ybar_mean)/ybar_std
    '''

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

    def init_phi(self, var_threshold: float = None):
        """
        Initialization of orthogonal basis, computed with singular value decomposition.
        """
        y_in = getattr(self, 'ybar_s', None)
        if y_in is None:
            y_in = self.y

        y, q = y_in, self.q
        n, p = self.n, self.p
        
        '''
        Replace y with ybar_s here.
             singvals, left_u, _ = tf.linalg.svd(self.ybar_s, full_matrices=False)
        '''

        singvals, left_u, _ = tf.linalg.svd(y, full_matrices=False)

        if (q is None) and (var_threshold is None):
            q = p
        elif (q is None) and (var_threshold is not None):
            cumvar = tf.cumsum(singvals ** 2) / tf.reduce_sum(singvals ** 2)
            q = int(tf.argmax(cumvar > var_threshold) + 1)

        assert left_u.shape[1] == min(n, p)
        singvals = singvals[:q]

        # Compute phi and diag_D
        phi = left_u[:, :q] * tf.sqrt(tf.cast(n, tf.float64)) / singvals
        diag_D = tf.reduce_sum(phi ** 2, axis=0)

        '''
        y_bar here for replication
        '''
        g = tf.matmul(phi, y, transpose_a=True)
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

        self.n = tf.constant(nx, tf.int32)
        self.d = tf.constant(d, tf.int32)
        self.p = tf.constant(p, tf.int32)
        return

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

    def fit(self, verbose=False):
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.loss, self.trainable_variables)
        return

    def loss(self):
        """
        Computes the loss based on the submethod.
        """
        if self.submethod == 'full':
            return self.neglpost()
        # elif self.submethod == 'elbo':
        #     return self.negelbo()
        # elif self.submethod == 'proflik':
        #     return self.negproflik()
        else:
            raise ValueError("Invalid submethod. Choices are 'full', 'elbo', or 'proflik'.")

    @tf.function
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
        self._ensure_replication()

        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        xk = self.x_unique                          # (n,d)
        ybar = self.ybar                            # (p,n)
        r = tf.cast(self.r, tf.float64)             # (n,)
        R = self.R                                  # (n,n)
        n = tf.cast(self.n, tf.float64)
        p = tf.cast(self.p, tf.float64)

        D = self.diag_D                             # (q,)
        phi = self.phi                              # (p,q)

        sigma_log = lsigma2s                        # (p,)
        sigma_inv = tf.exp(-sigma_log)              # Σ^{-1} diag
        sigma_inv_sqrt = tf.exp(-0.5 * sigma_log)   # Σ^{-1/2}

        nlp = tf.constant(0.0, tf.float64)

        # 0.5 * sum_i r_i * ybar_i^T Σ^{-1} ybar_i
        ybar_scaled = ybar * sigma_inv_sqrt[:, None]            # (p,n)
        col_sq = tf.reduce_sum(tf.square(ybar_scaled), axis=0)  # (n,)
        nlp += 0.5 * tf.reduce_sum(r * col_sq)

        # + (n/2) log|Σ|
        nlp += 0.5 * n * tf.reduce_sum(sigma_log)

        # - (p/2) log|R| = - (p/2) * sum_i log(r_i)
        nlp += -0.5 * p * tf.reduce_sum(tf.math.log(tf.cast(self.r, tf.float64)))

        pc = self.penalty_const
        reg = (pc['lLmb'] * tf.reduce_sum(tf.square(tf.math.log(self.lLmb))) +
               pc['lLmb0'] * (2.0 / n) * tf.reduce_sum(tf.square(self.lLmb0.unconstrained_variable))
               - tf.reduce_sum(tf.math.log(tf.math.log(self.lnugGPs) + 100.0))
              )

        bkSb_sum = tf.constant(0.0, tf.float64)
        logA_sum = tf.constant(0.0, tf.float64)

        sr = tf.sqrt(r)  # (n,)

        q_int = tf.cast(self.q, tf.int32)
        for k in range(q_int):
            Ck = Matern32(xk, xk, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])  # (n,n)

            # b_k = R * ybar^T * (Σ^{-1/2} phi_k)
            phi_k = phi[:, k]
            v_k = phi_k * sigma_inv_sqrt
            ytv = tf.linalg.matvec(tf.transpose(ybar), v_k)   # (n,)
            b_k = r * ytv

            # A = I + d_k * R^{1/2} C_k R^{1/2}
            d_k = D[k]
            Csr = Ck * sr[None, :]
            A = tf.eye(self.n, dtype=tf.float64) + d_k * (Csr * sr[:, None])

            LA = tf.linalg.cholesky(A)
            logdetA = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LA)))
            logA_sum += logdetA

            Cb = tf.linalg.matvec(Ck, b_k)
            u = tf.sqrt(d_k) * (sr * Cb)
            z = tf.linalg.cholesky_solve(LA, tf.expand_dims(u, -1))
            z = tf.squeeze(z, -1)
            Sb = Cb - tf.linalg.matvec(Ck, (tf.sqrt(d_k) * (sr * z)))

            bkSb_sum += tf.tensordot(b_k, Sb, axes=1)

        nlp += -0.5 * bkSb_sum
        nlp += 0.5 * logA_sum
        nlp += reg
        nlp /= n
        return nlp

    @tf.function
    def neglpost(self):
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

        # no calling kronecker
        # use woodbury for inversions of something like (dk Ck + R^{-1}???)^{-1}
        for k in range(q):
            Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

            # components necessary for Ck^{-1}
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
        nlp += (pc['lLmb'] * tf.reduce_sum(tf.square(tf.math.log(lLmb))) +
                pc['lLmb0'] * (2 / n) * tf.reduce_sum(tf.square(lLmb0.unconstrained_variable)))
        nlp += (-tf.reduce_sum(tf.math.log(tf.math.log(lnugGPs) + 100)))
        # nlp += (tf.reduce_sum(tf.math.log(lnugGPs - 100)))
        nlp /= tf.cast(n, tf.float64)
        return nlp

    def predict(self, x0, return_fullcov=False):
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


    def compute_aux_predictive_quantities(self):
        """
        Compute auxiliary quantities for predictions using full posterior approach.
        """
        # If replication is present, compute replicated auxiliaries, else original ones
        if hasattr(self, 'x_unique') and hasattr(self, 'ybar'):
            self._compute_aux_predictive_quantities_rep()
            return
        x = self.x
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()

        D = self.diag_D
        # B := Y @ Sigma^{-1/2} @ Phi
        B = tf.matmul(tf.transpose(self.y) / tf.sqrt(tf.exp(lsigma2s)), self.phi)

        '''
        Use averaged data in replicated model:
            - B = Ybar_s^T @ Σ^{-1/2} @ Φ                              
            - Set x = self.x_unique for all covariances below.
        '''

        CinvM = tf.zeros([self.q, self.n], dtype=tf.float64)
        Th = tf.zeros([self.q, self.n, self.n], dtype=tf.float64)

        for k in range(self.q):
            Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

            Wk, Uk = tf.linalg.eigh(Ck)

            # (I + D_k * C_k)^{-1}
            IpdkCkinv = tf.matmul(Uk, tf.matmul(tf.linalg.diag(1.0 / (1.0 + D[k] * Wk)), tf.transpose(Uk)))

            '''
            With replication the transform changes:
                S_k = (C_k^{-1} + d_k R)^{-1}                            
                T_k = C_k^{-1} - C_k^{-1} S_k C_k^{-1} for var
            '''

            CkinvMk = tf.linalg.matvec(IpdkCkinv, tf.transpose(B)[k])
            Thk = tf.matmul(Uk, tf.matmul(tf.linalg.diag(tf.sqrt((D[k] * Wk ** 2) / (Wk ** 2 + D[k] * Wk ** 3))),
                                          tf.transpose(Uk)))

            CinvM = tf.tensor_scatter_nd_update(CinvM, [[k]], tf.expand_dims(CkinvMk, axis=0))
            Th = tf.tensor_scatter_nd_update(Th, [[k]], tf.expand_dims(Thk, axis=0))

        self.CinvMs = CinvM
        self.Ths = Th

    def _compute_aux_predictive_quantities_rep(self):
        """
        Replication-aware auxiliaries:
          - CinvMs[k] = C_k^{-1} m_k = b_k - d_k * R @ m_k  (avoid explicit C^{-1})
          - Tks[k]    = C_k^{-1} - C_k^{-1} S_k C_k^{-1},   S_k = (C_k^{-1}+d_k R)^{-1}
        """
        self._ensure_replication()

        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        xk = self.x_unique
        ybar = self.ybar
        r = tf.cast(self.r, tf.float64)
        R = self.R

        D = self.diag_D
        phi = self.phi

        sigma_inv_sqrt = tf.exp(-0.5 * lsigma2s)

        q = tf.cast(self.q, tf.int32)
        n = tf.cast(self.n, tf.int32)

        CinvM = tf.zeros([q, n], dtype=tf.float64)
        Tks   = tf.zeros([q, n, n], dtype=tf.float64)

        for k in range(q):
            Ck = Matern32(xk, xk, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])  # (n,n)
            phi_k = phi[:, k]
            v_k = phi_k * sigma_inv_sqrt
            b_k = r * tf.linalg.matvec(tf.transpose(ybar), v_k)  # (n,)

            # m = S b via Woodbury in C-space:
            d_k = D[k]
            sr = tf.sqrt(r)
            Cb = tf.linalg.matvec(Ck, b_k)
            A = tf.eye(n, dtype=tf.float64) + d_k * ((Ck * sr[None, :]) * sr[:, None])  # I + d R^{1/2} C R^{1/2}
            LA = tf.linalg.cholesky(A)
            u = tf.sqrt(d_k) * (sr * Cb)
            z = tf.linalg.cholesky_solve(LA, tf.expand_dims(u, -1))
            z = tf.squeeze(z, -1)
            m_k = Cb - tf.linalg.matvec(Ck, (tf.sqrt(d_k) * (sr * z)))  # (n,)

            # C^{-1} m = b - d R m
            CinvM_k = b_k - d_k * tf.linalg.matvec(R, m_k)

            # T = C^{-1} - C^{-1} S C^{-1}
            LC = tf.linalg.cholesky(Ck)
            Id = tf.eye(n, dtype=tf.float64)
            invC = tf.linalg.cholesky_solve(LC, Id)                 # C^{-1}
            A_inv = tf.linalg.inv(invC + d_k * tf.cast(R, tf.float64))  # S
            Tk = invC - invC @ A_inv @ invC

            CinvM = tf.tensor_scatter_nd_update(CinvM, [[k]], [CinvM_k])
            Tks   = tf.tensor_scatter_nd_update(Tks,   [[k]], [Tk])

        self.CinvMs = CinvM
        self.Tks    = Tks
        self.Ths    = None

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
        x0 = (x0 - self.x_min) / (self.x_max - self.x_min)  # Standardize x0
        n0 = tf.shape(x0)[0]

        ghat = tf.zeros([self.q, n0], dtype=tf.float64)
        gvar = tf.zeros([self.q, n0], dtype=tf.float64)
        for k in range(self.q):
            c00k = Matern32(x0, x0, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k],
                            diag_only=True)  # Diagonal-only covariance
            c0k = Matern32(x0, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k],
                           diag_only=False)

            '''
            Replace x with self.x_unique.
            Mean/var structure stays the same, but uses:
                - T_k = C_k^{-1} - C_k^{-1} S_k C_k^{-1}                    
                - gvar_k = c00k - c0k @ T_k @ c0k^T (diagonal extracted)
            '''

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
            yfullpredcov = (tf.einsum('nij,njk->nik', CH,
                                     tf.transpose(CH, perm=[0, 2, 1])) +
                            tf.linalg.diag(tf.exp(lsigma2s)))
            yfullpredcov *= tf.square(self.ystd)
            return ypred, ypredvar, yconfvar, yfullpredcov

        return ypred, ypredvar, yconfvar

    def predict_rep(self, x0, return_fullcov=False):
        """
        Replication-aware prediction using Tks and CinvMs computed on unique inputs.
        Mean:  ghat_k(x0) = c0k @ (C^{-1} m_k) = c0k @ CinvMs[k]
        Var:   gvar_k(x0) = c00k - row_diag(c0k @ T_k @ c0k^T)
        """
        self._ensure_replication()
        need_aux = (self.Tks is None) or tf.reduce_any(tf.math.is_nan(self.CinvMs))
        if need_aux:
            self._compute_aux_predictive_quantities_rep()

        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
        phi = self.phi

        Xtrain = self.x_unique
        Tks = self.Tks
        CinvM = self.CinvMs

        x0 = self._verify_data_types(x0)
        x0 = (x0 - self.x_min) / (self.x_max - self.x_min)
        n0 = tf.shape(x0)[0]

        ghat = tf.zeros([self.q, n0], dtype=tf.float64)
        gvar = tf.zeros([self.q, n0], dtype=tf.float64)

        for k in range(self.q):
            c00k = Matern32(x0, x0,   llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k], diag_only=True)
            c0k  = Matern32(x0, Xtrain, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k], diag_only=False)

            ghat_k = tf.linalg.matvec(c0k, CinvM[k])

            Tk = Tks[k]
            v = tf.matmul(c0k, Tk)               
            quad = tf.reduce_sum(v * c0k, axis=1)
            gvar_k = c00k - quad

            ghat = tf.tensor_scatter_nd_update(ghat, [[k]], [ghat_k])
            gvar = tf.tensor_scatter_nd_update(gvar, [[k]], [gvar_k])

        psi = tf.transpose(phi) * tf.sqrt(tf.exp(lsigma2s))
        predmean = tf.matmul(psi, ghat, transpose_a=True)
        confvar  = tf.matmul(tf.transpose(gvar), tf.square(psi))
        predvar  = confvar + tf.exp(lsigma2s)

        ypred    = self.tx_y(predmean)
        yconfvar = tf.transpose(confvar) * tf.square(self.ystd)
        ypredvar = tf.transpose(predvar) * tf.square(self.ystd)

        if return_fullcov:
            CH = tf.sqrt(gvar)[..., tf.newaxis] * psi[tf.newaxis, ...]
            yfullpredcov = (tf.einsum('nij,njk->nik', CH, tf.transpose(CH, perm=[0, 2, 1])) +
                            tf.linalg.diag(tf.exp(lsigma2s)))
            yfullpredcov *= tf.square(self.ystd)
            return ypred, ypredvar, yconfvar, yfullpredcov

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
