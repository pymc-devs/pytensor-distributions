"""Matrix Normal distribution."""

from functools import partial

import pytensor.tensor as pt
from pytensor.tensor.linalg import solve_triangular

from pytensor_distributions.mvnormal import _logdet_from_cholesky

solve_lower = partial(solve_triangular, lower=True)


def _broadcast_mu(mu, rowcov, colcov):
    """Broadcast mu to the output shape implied by all parameters."""
    mu = pt.as_tensor(mu)
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)
    # Use zero-addition to trigger automatic broadcasting across batch dims
    row_zeros = pt.zeros(rowcov.shape[:-1])  # (..., m)
    col_zeros = pt.zeros(colcov.shape[:-1])  # (..., n)
    return mu + row_zeros[..., :, None] * col_zeros[..., None, :]


def mean(mu, rowcov, colcov):
    return _broadcast_mu(mu, rowcov, colcov)


def mode(mu, rowcov, colcov):
    return _broadcast_mu(mu, rowcov, colcov)


def median(mu, rowcov, colcov):
    return _broadcast_mu(mu, rowcov, colcov)


def var(mu, rowcov, colcov):
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)
    row_diag = pt.diagonal(rowcov, axis1=-2, axis2=-1)
    col_diag = pt.diagonal(colcov, axis1=-2, axis2=-1)
    return row_diag[..., :, None] * col_diag[..., None, :]


def std(mu, rowcov, colcov):
    return pt.sqrt(var(mu, rowcov, colcov))


def skewness(mu, rowcov, colcov):
    return pt.zeros_like(_broadcast_mu(mu, rowcov, colcov))


def kurtosis(mu, rowcov, colcov):
    return pt.zeros_like(_broadcast_mu(mu, rowcov, colcov))


def entropy(mu, rowcov, colcov):
    mu = pt.as_tensor(mu)
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)
    m = mu.shape[-2]
    n = mu.shape[-1]
    mn = m * n
    _, logdet_U = pt.linalg.slogdet(rowcov)
    _, logdet_V = pt.linalg.slogdet(colcov)
    return 0.5 * mn * pt.log(2 * pt.pi * pt.e) + 0.5 * n * logdet_U + 0.5 * m * logdet_V


def logpdf(X, mu, rowcov, colcov):
    X = pt.as_tensor(X)
    mu = pt.as_tensor(mu)
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)

    m = mu.shape[-2]
    n = mu.shape[-1]
    mn = m * n

    chol_row = pt.linalg.cholesky(rowcov, lower=True)
    chol_col = pt.linalg.cholesky(colcov, lower=True)

    logdet_row, _ = _logdet_from_cholesky(chol_row)
    logdet_col, _ = _logdet_from_cholesky(chol_col)

    delta = X - mu

    # Compute tr[V^-1 (X-M)^T U^-1 (X-M)] via Cholesky solves
    # Using vec identity: quadform = ||L_U^-1 delta L_V^-T||^2_F
    Y = solve_lower(chol_row, delta)  # L_U^-1 delta, shape (..., m, n)
    Z = solve_lower(chol_col, pt.swapaxes(Y, -1, -2))  # L_V^-1 (L_U^-1 delta)^T, shape (..., n, m)
    quadform = pt.sum(Z**2, axis=(-2, -1))

    log_norm = 0.5 * mn * pt.log(2 * pt.pi) + 0.5 * n * logdet_row + 0.5 * m * logdet_col

    return -0.5 * quadform - log_norm


def pdf(X, mu, rowcov, colcov):
    return pt.exp(logpdf(X, mu, rowcov, colcov))


def rvs(mu, rowcov, colcov, size=None, random_state=None):
    mu = pt.as_tensor(mu)
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)

    L_row = pt.linalg.cholesky(rowcov, lower=True)
    L_col = pt.linalg.cholesky(colcov, lower=True)

    if size is None:
        size = ()
    elif not isinstance(size, tuple):
        size = (size,)

    # Get the broadcast output shape from parameters
    target = _broadcast_mu(mu, rowcov, colcov)  # (..., m, n)
    base_shape = target.shape  # symbolic shape vector

    if size:
        full_shape = pt.concatenate([pt.as_tensor(size), base_shape])
    else:
        full_shape = base_shape

    Z = pt.random.normal(0, 1, size=full_shape, rng=random_state, return_next_rng=True)[1]
    return target + L_row @ Z @ pt.swapaxes(L_col, -1, -2)
