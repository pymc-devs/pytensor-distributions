import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_cont


def mean(lam):
    return 1.0 / lam


def mode(lam):
    shape = pt.broadcast_arrays(lam)[0]
    return pt.full_like(shape, 0.0)


def median(lam):
    return pt.log(2.0) / lam


def var(lam):
    return 1.0 / pt.square(lam)


def std(lam):
    return 1.0 / lam


def skewness(lam):
    shape = pt.broadcast_arrays(lam)[0]
    return pt.full_like(shape, 2.0)


def kurtosis(lam):
    shape = pt.broadcast_arrays(lam)[0]
    return pt.full_like(shape, 6.0)


def lmoment1(lam):
    return mean(lam)


def lmoment2(lam):
    shape = pt.broadcast_arrays(lam)[0]
    return pt.full_like(shape, 1 / lam * 0.5)


def lmoment3(lam):
    shape = pt.broadcast_arrays(lam)[0]
    return pt.full_like(shape, 1 / 3)


def lmoment4(lam):
    shape = pt.broadcast_arrays(lam)[0]
    return pt.full_like(shape, 1 / 6)


def entropy(lam):
    return 1.0 - pt.log(lam)


def cdf(x, lam):
    result = 1.0 - pt.exp(-lam * x)
    return cdf_bounds(result, x, 0.0, pt.inf)


def isf(x, lam):
    return ppf(1 - x, lam)


def pdf(x, lam):
    return pt.switch(pt.lt(x, 0.0), 0.0, lam * pt.exp(-lam * x))


def ppf(q, lam):
    result = -pt.log(1.0 - q) / lam
    return ppf_bounds_cont(result, q, 0.0, pt.inf)


def sf(x, lam):
    return pt.switch(pt.lt(x, 0.0), 1.0, pt.exp(-lam * x))


def rvs(lam, size=None, random_state=None):
    return pt.random.exponential(1.0 / lam, rng=random_state, size=size)


def logcdf(x, lam):
    return pt.switch(pt.lt(x, 0.0), -pt.inf, pt.log1p(-pt.exp(-lam * x)))


def logpdf(x, lam):
    return pt.switch(pt.lt(x, 0.0), -pt.inf, pt.log(lam) - lam * x)


def logsf(x, lam):
    return pt.switch(pt.lt(x, 0.0), 0.0, -lam * x)


def from_beta(beta):
    return 1 / beta


def to_beta(lam):
    return 1 / lam
