import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.math import gammaln

from pytensor_distributions.mvnormal import quaddist_chol


def mean(nu, mu, cov):
    mu = pt.broadcast_to(mu, cov.shape[:-1])
    return pt.switch(nu > 1, mu, pt.nan)


def mode(nu, mu, cov):
    return pt.broadcast_to(mu, cov.shape[:-1])


def median(nu, mu, cov):
    return pt.broadcast_to(mu, cov.shape[:-1])


def var(nu, mu, cov):
    diag_cov = pt.diagonal(cov, axis1=-2, axis2=-1)
    return pt.switch(nu > 2, nu / (nu - 2) * diag_cov, pt.inf)


def std(nu, mu, cov):
    return pt.sqrt(var(nu, mu, cov))


def skewness(nu, mu, cov):
    mu = pt.broadcast_to(mu, cov.shape[:-1])
    return pt.switch(nu > 3, 0.0, pt.nan)


def kurtosis(nu, mu, cov):
    mu = pt.broadcast_to(mu, cov.shape[:-1])
    k = cov.shape[-1]
    return pt.switch(nu > 4, 6 * (nu - 2) / ((k + 2) * (nu - 4)), pt.inf)


def entropy(nu, mu, cov):
    k = cov.shape[-1]
    _, logdet = pt.linalg.slogdet(cov)
    return (
        0.5 * logdet
        + 0.5 * k * pt.log(nu * pt.pi)
        + (k + nu) / 2 * pt.digamma((k + nu) / 2)
        - (k + nu) / 2 * pt.digamma(nu / 2)
        + gammaln(nu / 2)
        - gammaln((k + nu) / 2)
    )


def pdf(x, nu, mu, cov):
    return pt.exp(logpdf(x, nu, mu, cov))


def logpdf(x, nu, mu, cov):
    quaddist, logdet, _ = quaddist_chol(x, mu, cov)
    k = pt.as_tensor(x.shape[-1], dtype="floatX")

    norm = gammaln((nu + k) / 2.0) - gammaln(nu / 2.0) - 0.5 * k * pt.log(nu * np.pi)
    inner = -(nu + k) / 2.0 * pt.log1p(quaddist / nu)
    res = norm + inner - 0.5 * logdet

    return pt.switch(
        pt.bitwise_or(pt.any(pt.lt(x, -pt.inf)), pt.any(pt.gt(x, pt.inf))), -pt.inf, res
    )


def rvs(nu, mu, cov, size=None, random_state=None):
    mu = pt.broadcast_to(mu, cov.shape[:-1])
    next_rng, z = pt.random.multivariate_normal(
        pt.zeros_like(mu), cov, size=size, rng=random_state, return_next_rng=True
    )
    chi2 = pt.random.chisquare(
        nu, size=size if size is not None else 1, rng=next_rng, return_next_rng=True
    )[1]
    if size is None:
        chi2 = chi2[None]
    return mu + z / pt.sqrt(chi2[..., None] / nu)
