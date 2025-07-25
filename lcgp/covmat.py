import tensorflow as tf
tf.keras.backend.set_floatx('float64')


def Matern32(x1, x2, llmb, llmb0, lnug, diag_only: bool = False):
    """
    Returns the Matern 3/2 covariance matrix.

    :param x1: input 1 of size (number of inputs in x1, dimension of input)
    :param x2: input 2 of size (number of inputs in x2, dimension of input)
    :param llmb: log-lengthscale hyperparameter for each dimension
    :param llmb0: log-scale hyperparameter
    :param lnug: parameter to tune the nugget, nugget = exp(lnug) / (1 + exp(lnug))
    :param diag_only: returns diagonal of covariance matrix if True. Default to False.
    :return: covariance matrix of size (n1, n2)
    """
    # assumes tensors are supplied
    assert x1.ndim == 2, 'input x1 should be 2-dimensional, (n_param, dim_param)'
    assert x2.ndim == 2, 'input x2 should be 2-dimensional, (n_param, dim_param)'
    assert x1.shape[1] == x2.shape[1], 'the dim_param of input x1 and x2 should be the same.'
    d = x1.shape[1]

    if diag_only:
        assert tf.reduce_all(tf.keras.ops.isclose(x1, x2)), \
            'diag_only should only be called ' \
            'when x1 and x2 are identical.'
        c = tf.ones(x1.shape[0], dtype=tf.float64)  # / (1 + tf.exp(llmb[-1]))
        return llmb0 * c

    else:
        V = tf.zeros((x1.shape[0], x2.shape[0]), dtype=tf.float64)
        C0 = tf.ones((x1.shape[0], x2.shape[0]), dtype=tf.float64)  # / (1 + tf.exp(llmb[-1]))

        x1scal = x1 / llmb  # [:-1])
        x2scal = x2 / llmb  # [:-1])
        for j in range(d):
            S = tf.abs(tf.reshape(x1scal[:, j], (-1, 1)) - x2scal[:, j])  # outer diff
            C0 *= (1 + S)
            V -= S

        C0 *= tf.exp(V)
        # C0 += tf.exp(llmb[-1]) / (1 + tf.exp(llmb[-1]))

        nug = lnug / (1 + lnug)
        if x1.shape != x2.shape:
            C = (1 - nug) * C0
        elif tf.reduce_all(tf.keras.ops.equal(x1, x2)):
            C = (1 - nug) * C0 + nug * tf.eye(x1.shape[0], dtype=tf.float64)
        else:
            C = (1 - nug) * C0

        return llmb0 * C


def Matern32_sp(x, xi, llmb, llmb0, lnug):  # assuming x1 = x2 = theta
    '''
    Returns the Nystr{\"o}m approximation of a covariance matrix,
    its inverse, and the log of its determinant.

    :param x: input of size (number of inputs, dimension of input)
    :param xi: inducing inputs of size (number of inducing inputs, dimension of input)
    :param llmb: log-lengthscale hyperparameter for each dimension
    :param llmb0: log-scale hyperparameter
    :param lnug: parameter to tune the nugget, nugget = exp(lnug) / (1 + exp(lnug))
    :return: a covariance matrix of size (n, n)
    '''

    c_full_i = Matern32(x, xi, llmb=llmb, llmb0=llmb0, lnug=lnug)
    C_i = Matern32(xi, xi, llmb=llmb, llmb0=llmb0, lnug=lnug)
    C_full_diag = Matern32(x, x, llmb=llmb, llmb0=llmb0, lnug=lnug, diag_only=True)

    Wi, Ui = tf.linalg.eigh(C_i)
    Ciinvh = Ui / Wi.sqrt()

    Crh = c_full_i @ Ciinvh

    diag = C_full_diag - (Crh ** 2).sum(1)  #
    Delta_inv_diag = 1 / diag

    R = C_i + (c_full_i.T * Delta_inv_diag) @ c_full_i

    WR, UR = tf.linalg.eigh(R)
    Rinvh = UR / WR.abs().sqrt()

    Q = (Delta_inv_diag * c_full_i.T).T
    Q_Rinvh = Q @ Rinvh

    logdet_C_sp = tf.log(WR.abs()).sum() - tf.log(Wi).sum() \
                  + tf.log(diag).sum()
    return Delta_inv_diag, Q, Rinvh, Q_Rinvh, logdet_C_sp, c_full_i, C_i
