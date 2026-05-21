import numpy as np
import pytensor.tensor as pt

from pytensor_distributions.helper import (
    cdf_bounds,
    ppf_bounds_cont,
)
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.normal import ppf as normal_ppf


def _logit(x):
    return pt.log(x) - pt.log1p(-x)


def _ghq_moments(mu, sigma, order=1, mean_val=None, n_points=70):
    """
    Compute moments of the logit-normal using Gauss-Hermite quadrature.

    Based on https://en.wikipedia.org/wiki/Logit-normal_distribution#Moments
    but using Gauss-Hermite quadrature for better accuracy.

    Parameters
    ----------
    mu : tensor
        Mean of underlying normal distribution
    sigma : tensor
        Standard deviation of underlying normal distribution
    order : int
        Order of the moment
    mean_val : tensor, optional
        If provided, compute central moment around this mean
    n_points : int
        Number of Gauss–Hermite nodes

    Returns
    -------
    tensor
        Estimated moment
    """
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_points)
    gh_x = pt.as_tensor_variable(gh_x)
    gh_w = pt.as_tensor_variable(gh_w)

    broadcast_shape = pt.broadcast_arrays(mu, sigma)[0]

    gh_x_bc = gh_x.reshape((-1,) + (1,) * broadcast_shape.ndim)
    gh_w_bc = gh_w.reshape((-1,) + (1,) * broadcast_shape.ndim)

    z = pt.sqrt(2.0) * sigma * gh_x_bc + mu
    x_vals = pt.sigmoid(z)

    if mean_val is not None:
        integrand = (x_vals - mean_val) ** order
    else:
        integrand = x_vals**order

    result = pt.sum(gh_w_bc * integrand, axis=0) / pt.sqrt(pt.pi)

    return result


def mean(mu, sigma):
    return _ghq_moments(mu, sigma, order=1)


def mode(mu, sigma):
    return pt.sigmoid(mu)


def median(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, pt.sigmoid(mu))


def var(mu, sigma):
    mean_val = _ghq_moments(mu, sigma, order=1)
    return _ghq_moments(mu, sigma, order=2, mean_val=mean_val)


def std(mu, sigma):
    return pt.sqrt(var(mu, sigma))


def skewness(mu, sigma):
    mean_val = _ghq_moments(mu, sigma, order=1)
    variance = _ghq_moments(mu, sigma, order=2, mean_val=mean_val)
    third_central = _ghq_moments(mu, sigma, order=3, mean_val=mean_val)
    return third_central / (pt.sqrt(variance) ** 3)


def kurtosis(mu, sigma):
    mean_val = _ghq_moments(mu, sigma, order=1)
    variance = _ghq_moments(mu, sigma, order=2, mean_val=mean_val)
    fourth_central = _ghq_moments(mu, sigma, order=4, mean_val=mean_val)
    return fourth_central / (variance**2) - 3


def lmoment1(mu, sigma):
    return mean(mu, sigma)


def lmoment2(mu, sigma):
    return _lmoments(ppf, mu, sigma, r=2)


def lmoment3(mu, sigma):
    return _lmoments(ppf, mu, sigma, r=3)


def lmoment4(mu, sigma):
    return _lmoments(ppf, mu, sigma, r=4)


def entropy(mu, sigma):
    gh_x, gh_w = np.polynomial.hermite.hermgauss(70)
    gh_x = pt.as_tensor_variable(gh_x)
    gh_w = pt.as_tensor_variable(gh_w)

    broadcast_shape = pt.broadcast_arrays(mu, sigma)[0]

    gh_x_bc = gh_x.reshape((-1,) + (1,) * broadcast_shape.ndim)
    gh_w_bc = gh_w.reshape((-1,) + (1,) * broadcast_shape.ndim)

    z = pt.sqrt(2.0) * sigma * gh_x_bc + mu
    x_vals = pt.sigmoid(z)

    integrand = -logpdf(x_vals, mu, sigma)

    result = pt.sum(gh_w_bc * integrand, axis=0) / pt.sqrt(pt.pi)

    return result


def pdf(x, mu, sigma):
    return pt.exp(logpdf(x, mu, sigma))


def logpdf(x, mu, sigma):
    logit_x = _logit(x)
    return pt.switch(
        pt.or_(pt.le(x, 0), pt.ge(x, 1)),
        -pt.inf,
        -0.5 * ((logit_x - mu) / sigma) ** 2
        - pt.log(sigma)
        - 0.5 * pt.log(2 * pt.pi)
        - pt.log(x)
        - pt.log1p(-x),
    )


def cdf(x, mu, sigma):
    logit_x = _logit(x)
    prob = 0.5 * (1 + pt.erf((logit_x - mu) / (sigma * pt.sqrt(2))))
    return cdf_bounds(prob, x, 0, 1)


def logcdf(x, mu, sigma):
    logit_x = _logit(x)
    z = (logit_x - mu) / sigma
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        pt.switch(
            pt.ge(x, 1),
            0.0,
            pt.switch(
                pt.lt(z, -1.0),
                pt.log(pt.erfcx(-z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
                pt.log1p(-pt.erfc(z / pt.sqrt(2.0)) / 2.0),
            ),
        ),
    )


def sf(x, mu, sigma):
    return pt.exp(logsf(x, mu, sigma))


def logsf(x, mu, sigma):
    logit_x = _logit(x)
    z = (logit_x - mu) / sigma
    return pt.switch(
        pt.le(x, 0),
        0.0,
        pt.switch(
            pt.ge(x, 1),
            -pt.inf,
            pt.switch(
                pt.gt(z, 1.0),
                pt.log(pt.erfcx(z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
                pt.log1p(-0.5 * (1 + pt.erf(z / pt.sqrt(2.0)))),
            ),
        ),
    )


def ppf(q, mu, sigma):
    return ppf_bounds_cont(pt.sigmoid(normal_ppf(q, mu, sigma)), q, 0, 1)


def isf(q, mu, sigma):
    return ppf(1 - q, mu, sigma)


def rvs(mu, sigma, size=None, random_state=None):
    return pt.sigmoid(pt.random.normal(mu, sigma, rng=random_state, size=size))
