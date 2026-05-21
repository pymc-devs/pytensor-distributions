import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_disc
from pytensor_distributions.lmoments import _lmoments


def mean(p):
    return 1 / p


def mode(p):
    return pt.ones_like(p)


def median(p):
    return ppf(0.5, p)


def var(p):
    return (1 - p) / p**2


def std(p):
    return pt.sqrt(var(p))


def skewness(p):
    return (2 - p) / pt.sqrt(1 - p)


def kurtosis(p):
    return 6 + (p**2) / (1 - p)


def lmoment1(p):
    return mean(p)


def lmoment2(p):
    return (1 - p) / (p * (2 - p))


def lmoment3(p):
    return _lmoments(ppf, p, r=3)


def lmoment4(p):
    return _lmoments(ppf, p, r=4)


def entropy(p):
    return (-(1 - p) * pt.log(1 - p) - p * pt.log(p)) / p


def pdf(x, p):
    return pt.exp(logpdf(x, p))


def cdf(x, p):
    k = pt.floor(x)
    prob = -pt.expm1(pt.log1p(-p) * k)
    return cdf_bounds(prob, x, 1, pt.inf)


def ppf(q, p):
    vals = pt.ceil(pt.log1p(-q) / pt.log1p(-p))
    temp = cdf(vals - 1, p)
    result = pt.where(pt.and_(pt.ge(temp, q), pt.gt(vals, 0)), vals - 1, vals)
    return ppf_bounds_disc(result, q, 1, pt.inf)


def sf(x, p):
    return 1.0 - cdf(x, p)


def isf(q, p):
    return ppf(1.0 - q, p)


def rvs(p, size=None, random_state=None):
    return pt.random.geometric(p, size=size, rng=random_state)


def logpdf(x, p):
    return pt.switch(pt.lt(x, 1), -pt.inf, (x - 1) * pt.log1p(-p) + pt.log(p))


def logcdf(x, p):
    return pt.log(cdf(x, p))


def logsf(x, p):
    return pt.log1p(-cdf(x, p))
