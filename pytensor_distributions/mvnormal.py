from functools import partial

import pytensor.tensor as pt
from pytensor.tensor.linalg import solve_triangular

solve_lower = partial(solve_triangular, lower=True)


def _logdet_from_cholesky(chol):
    """Compute log determinant from Cholesky factor and check positive definiteness."""
    diag = pt.diagonal(chol, axis1=-2, axis2=-1)
    logdet = pt.sum(pt.log(diag), axis=-1) * 2
    posdef = pt.all(diag > 0, axis=-1)
    return logdet, posdef


def quaddist_chol(value, mu, cov):
    """Compute (x - mu).T @ Sigma^-1 @ (x - mu) and the logdet of Sigma."""
    if value.ndim == 0:
        raise ValueError("Value can't be a scalar")
    if value.ndim == 1:
        onedim = True
        value = value[None, :]
    else:
        onedim = False

    chol_cov = pt.linalg.cholesky(cov, lower=True)
    logdet, posdef = _logdet_from_cholesky(chol_cov)

    delta = value - mu
    delta_trans = solve_lower(chol_cov, delta, b_ndim=1)
    quaddist = (delta_trans**2).sum(axis=-1)

    if onedim:
        return quaddist[0], logdet, posdef
    else:
        return quaddist, logdet, posdef


def mean(mu, cov):
    return pt.broadcast_to(mu, cov.shape[:-1])


def mode(mu, cov):
    return pt.broadcast_to(mu, cov.shape[:-1])


def median(mu, cov):
    return pt.broadcast_to(mu, cov.shape[:-1])


def var(mu, cov):
    return pt.diagonal(cov, axis1=-2, axis2=-1)


def std(mu, cov):
    return pt.sqrt(var(mu, cov))


def skewness(mu, cov):
    mu = pt.broadcast_to(mu, cov.shape[:-1])
    return pt.zeros_like(mu)


def kurtosis(mu, cov):
    mu = pt.broadcast_to(mu, cov.shape[:-1])
    return pt.zeros_like(mu)


def entropy(mu, cov):
    k = cov.shape[-1]
    _, logdet = pt.linalg.slogdet(cov)
    return 0.5 * (k * pt.log(2 * pt.pi * pt.e) + logdet)


def pdf(x, mu, cov):
    return pt.exp(logpdf(x, mu, cov))


def logpdf(x, mu, cov):
    quaddist, logdet, _ = quaddist_chol(x, mu, cov)
    k = pt.as_tensor(x.shape[-1], dtype="floatX")
    return -0.5 * (k * pt.log(2 * pt.pi) + logdet + quaddist)


def rvs(mu, cov, size=None, random_state=None):
    mu = pt.broadcast_to(mu, cov.shape[:-1])
    return pt.random.multivariate_normal(
        mu, cov, size=size, rng=random_state, return_next_rng=True
    )[1]
