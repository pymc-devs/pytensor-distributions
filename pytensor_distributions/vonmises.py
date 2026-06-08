import pytensor.tensor as pt

from pytensor_distributions.helper import von_mises_cdf
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf


def mean(mu, kappa):
    shape = pt.broadcast_arrays(mu, kappa)[0]
    return pt.full_like(shape, mu)


def mode(mu, kappa):
    shape = pt.broadcast_arrays(mu, kappa)[0]
    return pt.full_like(shape, mu)


def median(mu, kappa):
    shape = pt.broadcast_arrays(mu, kappa)[0]
    return pt.full_like(shape, mu)


def var(mu, kappa):
    return 1 - pt.ive(1, kappa) / pt.ive(0, kappa)


def std(mu, kappa):
    return pt.sqrt(var(mu, kappa))


def skewness(mu, kappa):
    shape = pt.broadcast_arrays(mu, kappa)[0]
    return pt.full_like(shape, 0.0)


def kurtosis(mu, kappa):
    shape = pt.broadcast_arrays(mu, kappa)[0]
    return pt.full_like(shape, 0.0)


def lmoment1(mu, kappa):
    return mean(mu, kappa)


def lmoment2(mu, kappa):
    return _lmoments(ppf, mu, kappa, r=2)


def lmoment3(mu, kappa):
    shape = pt.broadcast_arrays(mu, kappa)[0]
    return pt.full_like(shape, 0.0)


def lmoment4(mu, kappa):
    return _lmoments(ppf, mu, kappa, r=4)


def entropy(mu, kappa):
    return pt.log(2 * pt.pi * pt.ive(0, kappa)) + kappa * var(mu, kappa)


def pdf(x, mu, kappa):
    return pt.exp(logpdf(x, mu, kappa))


def logpdf(x, mu, kappa):
    return kappa * (pt.cos(x - mu) - 1) - pt.log(2 * pt.pi) - pt.log(pt.ive(0, kappa))


def rvs(mu, kappa, size=None, random_state=None):
    return pt.random.vonmises(mu, kappa, rng=random_state, size=size, return_next_rng=True)[1]


def cdf(x, mu, kappa):
    return von_mises_cdf(x, mu, kappa)


def logcdf(x, mu, kappa):
    return pt.log(cdf(x, mu, kappa))


def sf(x, mu, kappa):
    return 1.0 - cdf(x, mu, kappa)


def logsf(x, mu, kappa):
    return pt.log1p(-cdf(x, mu, kappa))


def isf(q, mu, kappa):
    return ppf(1 - q, mu, kappa)


def ppf(q, mu, kappa):
    return find_ppf(q, mean(mu, kappa), -pt.inf, pt.inf, von_mises_cdf, pdf, mu, kappa)
