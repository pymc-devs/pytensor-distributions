import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(sigma):
    return sigma * pt.sqrt(2 / pt.pi)


def mode(sigma):
    return pt.zeros_like(sigma)


def median(sigma):
    return sigma * pt.sqrt(2) * pt.erfinv(0.5)


def var(sigma):
    return sigma**2 * (1 - 2 / pt.pi)


def std(sigma):
    return sigma * pt.sqrt(1 - 2 / pt.pi)


def skewness(sigma):
    return pt.full_like(sigma, 0.9952717)


def kurtosis(sigma):
    return pt.full_like(sigma, 0.8691773)


def lmoment1(sigma):
    return mean(sigma)


def lmoment2(sigma):
    # sigma * (2 - sqrt(2)) / sqrt(pi)
    return sigma * 0.3304946062926472


def lmoment3(sigma):
    return _lmoments(ppf, sigma, r=3)


def lmoment4(sigma):
    return _lmoments(ppf, sigma, r=4)


def entropy(sigma):
    return 0.5 * pt.log(pt.pi * sigma**2 / 2) + 0.5


def cdf(x, sigma):
    return pt.where(pt.lt(x, 0), 0.0, pt.erf(x / (sigma * pt.sqrt(2))))


def logcdf(x, sigma):
    return pt.where(pt.lt(x, 0), -pt.inf, pt.log(pt.erf(x / (sigma * pt.sqrt(2)))))


def isf(x, sigma):
    return ppf(1 - x, sigma)


def pdf(x, sigma):
    return pt.exp(logpdf(x, sigma))


def ppf(q, sigma):
    x_vals = sigma * pt.sqrt(2) * pt.erfinv(q)
    return ppf_bounds_cont(x_vals, q, 0, pt.inf)


def sf(x, sigma):
    return 1 - cdf(x, sigma)


def rvs(sigma, size=None, random_state=None):
    return pt.abs(pt.random.normal(0, sigma, rng=random_state, size=size))


def logpdf(x, sigma):
    return pt.where(
        pt.lt(x, 0), -pt.inf, pt.log(pt.sqrt(2 / pt.pi)) - pt.log(sigma) - 0.5 * (x / sigma) ** 2
    )


def logsf(x, sigma):
    return pt.log1mexp(logcdf(x, sigma))
