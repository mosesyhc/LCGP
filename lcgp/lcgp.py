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
        """
        super().__init__()
        self.verbose = verbose
        self.robust_mean = robust_mean
        self.x = self._verify_data_types(x)
        self.y = self._verify_data_types(y)

        self.method = 'LCGP'
        self.submethod = submethod
        self.submethod_loss_map = {'full': self.neglpost,
                                   # 'elbo': self.negelbo,
                                   # 'proflik': self.negproflik
                                   }
        self.submethod_predict_map = {'full': self.predict_full,
                                      # 'elbo': self.predict_elbo,
                                      # 'proflik': self.predict_proflik
                                      }

        self.parameter_clamp_flag = parameter_clamp_flag

        if (q is not None) and (var_threshold is not None):
            raise ValueError('Include only q or var_threshold but not both.')
        self.q = q
        self.var_threshold = var_threshold

        # standardize x to unit hypercube
        self.x, self.x_min, self.x_max, self.x_orig, self.xnorm = \
            self.init_standard_x(self.x)
        # standardize y
        self.y, self.ymean, self.ystd, self.y_orig = self.init_standard_y(self.y)

        # placeholders for variables
        self.n, self.d, self.p = 0., 0., 0.
        # verify that input and output dimensions match
        # sets n, d, and p
        self.verify_dim(self.y, self.x)

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
        self.CinvMs = tf.fill([self.q, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Ths = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Th_hats = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))
        self.Cinvhs = tf.fill([self.q, self.n, self.n], tf.constant(float('nan'), dtype=tf.float64))

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
        y, q = self.y, self.q
        n, p = self.n, self.p

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
        nlp += (pc['lLmb'] * tf.reduce_sum(tf.square(tf.math.log(lLmb))) +
                pc['lLmb0'] * (2 / n) * tf.reduce_sum(tf.square(lLmb0.unconstrained_variable)))
        nlp += (-tf.reduce_sum(tf.math.log(tf.math.log(lnugGPs) + 100)))
        # nlp += (tf.reduce_sum(tf.math.log(lnugGPs - 100)))
        nlp /= tf.cast(n, tf.float64)
        return nlp

    # def negelbo(self):
    #     n = self.n
    #     x = self.x
    #     y = self.y
    #     pc = self.penalty_const
    #
    #     lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
    #     B = tf.matmul(tf.transpose(y / tf.sqrt(tf.exp(lsigma2s))), self.phi)
    #     D = self.diag_D
    #     phi = self.phi
    #
    #     psi = tf.transpose(phi) * tf.sqrt(tf.exp(lsigma2s))
    #
    #     M = tf.zeros([self.q, n], dtype=tf.float64)
    #
    #     negelbo = tf.constant(0., dtype=tf.float64)
    #     for k in range(self.q):
    #         Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])
    #
    #         Wk, Uk = tf.linalg.eigh(Ck)
    #         dkInpCkinv = tf.matmul(Uk, tf.matmul(tf.linalg.diag(1 / Wk), tf.transpose(Uk))) + \
    #                      D[k] * tf.eye(n, dtype=tf.float64)
    #
    #         # (dk * I + Ck^{-1})^{-1}
    #         dkInpCkinv_inv = tf.matmul(Uk, tf.matmul(tf.linalg.diag(1 / (D[k] + 1 / Wk)), tf.transpose(Uk)))
    #         Mk = tf.linalg.matvec(dkInpCkinv_inv, tf.transpose(B)[k])
    #         Vk = 1 / tf.linalg.diag_part(dkInpCkinv)
    #
    #         CkinvhMk = tf.linalg.matvec(tf.matmul(tf.matmul(Uk, tf.linalg.diag(1 / tf.sqrt(Wk))), tf.transpose(Uk)),
    #                                     Mk)
    #
    #         M = tf.tensor_scatter_nd_update(M, [[k]], tf.expand_dims(Mk, axis=0))
    #
    #         negelbo += 0.5 * tf.reduce_sum(tf.math.log(Wk))
    #         negelbo += 0.5 * tf.reduce_sum(tf.square(CkinvhMk))
    #         negelbo -= 0.5 * tf.reduce_sum(tf.math.log(Vk))
    #         negelbo += 0.5 * tf.reduce_sum(
    #             Vk * D[k] * tf.linalg.diag_part(tf.matmul(Uk, tf.matmul(tf.linalg.diag(1 / Wk), tf.transpose(Uk)))))
    #
    #     resid = (tf.transpose(y) - tf.matmul(tf.transpose(M), psi)) / tf.sqrt(tf.exp(lsigma2s))
    #
    #     negelbo += 0.5 * tf.reduce_sum(tf.square(resid))
    #     negelbo += n / 2 * tf.reduce_sum(lsigma2s)
    #
    #     # Regularization
    #     negelbo += pc['lLmb'] * tf.reduce_sum(tf.square(lLmb)) + \
    #                pc['lLmb0'] * (2 / n) * tf.reduce_sum(tf.square(lLmb0))
    #     negelbo += -tf.reduce_sum(tf.math.log(lnugGPs + 100))
    #
    #     negelbo /= tf.cast(n, tf.float64)
    #
    #     return negelbo
    #
    # def negproflik(self):
    #     lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()
    #     x = self.x
    #     y = self.y
    #
    #     pc = self.penalty_const
    #
    #     n = self.n
    #     q = self.q
    #     D = self.diag_D
    #     phi = self.phi
    #     psi = tf.transpose(phi) * tf.sqrt(tf.exp(lsigma2s))
    #
    #     B = tf.matmul(tf.transpose(y / tf.sqrt(tf.exp(lsigma2s))), self.phi)
    #     G = tf.zeros([self.q, n], dtype=tf.float64)
    #
    #     negproflik = tf.constant(0., dtype=tf.float64)
    #
    #     for k in range(q):
    #         Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])
    #         Wk, Uk = tf.linalg.eigh(Ck)
    #
    #         dkInpCkinv_inv = tf.matmul(Uk, tf.matmul(tf.linalg.diag(1 / (D[k] + 1 / Wk)), tf.transpose(Uk)))
    #         Gk = tf.matmul(dkInpCkinv_inv, tf.transpose(B)[k])
    #
    #         CkinvhGk = tf.matmul(tf.matmul(Uk, tf.linalg.diag(1 / Wk)), tf.transpose(Uk)) @ Gk
    #
    #         G = tf.tensor_scatter_nd_update(G, [[k]], tf.expand_dims(Gk, axis=0))
    #
    #         negproflik += 0.5 * tf.reduce_sum(tf.math.log(Wk))
    #         negproflik += 0.5 * tf.reduce_sum(tf.square(CkinvhGk))
    #
    #     resid = (tf.transpose(y) - tf.matmul(tf.transpose(G), psi)) / tf.sqrt(tf.exp(lsigma2s))
    #
    #     negproflik += 0.5 * tf.reduce_sum(tf.square(resid))
    #     negproflik += n / 2 * tf.reduce_sum(lsigma2s)
    #
    #     negproflik += pc['lLmb'] * tf.reduce_sum(tf.square(lLmb)) + \
    #                   pc['lLmb0'] * (2 / n) * tf.reduce_sum(tf.square(lLmb0))
    #     negproflik += -tf.reduce_sum(tf.math.log(lnugGPs + 100))
    #
    #     negproflik /= tf.cast(n, tf.float64)
    #     return negproflik


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
        x = self.x
        lLmb, lLmb0, lsigma2s, lnugGPs = self.get_param()

        D = self.diag_D
        # B := Y @ Sigma^{-1/2} @ Phi
        B = tf.matmul(tf.transpose(self.y) / tf.sqrt(tf.exp(lsigma2s)), self.phi)

        CinvM = tf.zeros([self.q, self.n], dtype=tf.float64)
        Th = tf.zeros([self.q, self.n, self.n], dtype=tf.float64)

        for k in range(self.q):
            Ck = Matern32(x, x, llmb=lLmb[k], llmb0=lLmb0[k], lnug=lnugGPs[k])

            Wk, Uk = tf.linalg.eigh(Ck)

            # (I + D_k * C_k)^{-1}
            IpdkCkinv = tf.matmul(Uk, tf.matmul(tf.linalg.diag(1.0 / (1.0 + D[k] * Wk)), tf.transpose(Uk)))

            CkinvMk = tf.linalg.matvec(IpdkCkinv, tf.transpose(B)[k])
            Thk = tf.matmul(Uk, tf.matmul(tf.linalg.diag(tf.sqrt((D[k] * Wk ** 2) / (Wk ** 2 + D[k] * Wk ** 3))),
                                          tf.transpose(Uk)))

            CinvM = tf.tensor_scatter_nd_update(CinvM, [[k]], tf.expand_dims(CkinvMk, axis=0))
            Th = tf.tensor_scatter_nd_update(Th, [[k]], tf.expand_dims(Thk, axis=0))

        self.CinvMs = CinvM
        self.Ths = Th

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

    # def predict_elbo(self, x0, return_fullcov=False):
    #     pass
    #
    # def predict_proflik(self, x0, return_fullcov=False):
    #     pass


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
        # if self.parameter_clamp_flag:
        #     lLmb, lLmb0, lsigma2s, lnugGPs = \
        #         self.parameter_clamp(lLmb=self.lLmb, lLmb0=self.lLmb0,
        #                              lsigma2s=self.lsigma2s, lnugs=self.lnugGPs)
        # else:
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
