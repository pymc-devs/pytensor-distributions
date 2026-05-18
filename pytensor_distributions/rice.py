import pytensor.tensor as pt

from pytensor_distributions.helper import (
    cdf_bounds,
    continuous_entropy,
    continuous_kurtosis,
    continuous_mode,
    continuous_skewness,
    marcum_q1_complement,
)
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf


def _laguerre_half(q):
    """
    Compute L_{1/2}(-q) where L_{1/2} is the Laguerre polynomial of order 1/2.

    Uses scaled Bessel functions for numerical stability:
    L_{1/2}(-q) = (1 + q) * I_0(q/2) * e^{-q/2} + q * I_1(q/2) * e^{-q/2}
               = (1 + q) * ive(0, q/2) + q * ive(1, q/2)
    """
    half_q = q / 2
    return (1 + q) * pt.ive(0, half_q) + q * pt.ive(1, half_q)


def mean(nu, sigma):
    """Mean of Rice distribution using closed-form Laguerre polynomial expression."""
    q = nu**2 / (2 * sigma**2)
    return sigma * pt.sqrt(pt.pi / 2) * _laguerre_half(q)


def mode(nu, sigma):
    return continuous_mode(_lower_bound(), _upper_bound(nu, sigma), logpdf, nu, sigma)


def median(nu, sigma):
    return ppf(0.5, nu, sigma)


def var(nu, sigma):
    """Variance of Rice distribution using closed-form Laguerre polynomial expression."""
    q = nu**2 / (2 * sigma**2)
    L_half = _laguerre_half(q)
    return 2 * sigma**2 + nu**2 - (pt.pi * sigma**2 / 2) * L_half**2


def std(nu, sigma):
    return pt.sqrt(var(nu, sigma))


def skewness(nu, sigma):
    return continuous_skewness(_lower_bound(), _upper_bound(nu, sigma), logpdf, nu, sigma)


def kurtosis(nu, sigma):
    return continuous_kurtosis(_lower_bound(), _upper_bound(nu, sigma), logpdf, nu, sigma)


def lmoment1(nu, sigma):
    return mean(nu, sigma)


def lmoment2(nu, sigma):
    return _lmoments(ppf, nu, sigma, r=2)


def lmoment3(nu, sigma):
    return _lmoments(ppf, nu, sigma, r=3)


def lmoment4(nu, sigma):
    return _lmoments(ppf, nu, sigma, r=4)


def entropy(nu, sigma):
    return continuous_entropy(_lower_bound(), _upper_bound(nu, sigma), logpdf, nu, sigma)


def pdf(x, nu, sigma):
    return pt.exp(logpdf(x, nu, sigma))


def logpdf(x, nu, sigma):
    x = pt.as_tensor_variable(x)
    sigma2 = sigma**2
    z = x * nu / sigma2

    result = (
        pt.log(x) - 2 * pt.log(sigma) - (x**2 + nu**2) / (2 * sigma2) + pt.log(pt.ive(0, z)) + z
    )

    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        pt.switch(pt.isinf(x), pt.nan, result),
    )


def cdf(x, nu, sigma):
    prob = marcum_q1_complement(nu / sigma, x / sigma)
    return cdf_bounds(prob, x, 0, pt.inf)


def logcdf(x, nu, sigma):
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        pt.log(cdf(x, nu, sigma)),
    )


def sf(x, nu, sigma):
    return 1.0 - cdf(x, nu, sigma)


def logsf(x, nu, sigma):
    return pt.log(sf(x, nu, sigma))


def ppf(q, nu, sigma):
    return find_ppf(q, mean(nu, sigma), 0, pt.inf, cdf, pdf, nu, sigma)


def isf(q, nu, sigma):
    return ppf(1.0 - q, nu, sigma)


def rvs(nu, sigma, size=None, random_state=None):
    next_rng, x = pt.random.normal(nu, sigma, size=size, rng=random_state).owner.outputs
    y = pt.random.normal(0, sigma, size=size, rng=next_rng)
    return pt.sqrt(x**2 + y**2)


def _lower_bound():
    return 1e-10


def _upper_bound(nu, sigma):
    return nu + 10 * sigma
