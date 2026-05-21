import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, discrete_entropy, ppf_bounds_disc


def mean(lower, upper):
    return (upper + lower) / 2


def mode(lower, upper):
    lower_b, _ = pt.broadcast_arrays(lower, upper)
    return pt.full_like(lower_b, pt.nan)


def median(lower, upper):
    return pt.floor((upper + lower) / 2)


def var(lower, upper):
    n = upper - lower + 1
    return (n**2 - 1) / 12


def std(lower, upper):
    return pt.sqrt(var(lower, upper))


def skewness(lower, upper):
    return pt.zeros_like(lower)


def kurtosis(lower, upper):
    n = upper - lower + 1
    return -(6 * (n**2 + 1)) / (5 * (n**2 - 1))


def lmoment1(lower, upper):
    return mean(lower, upper)


def lmoment2(lower, upper):
    n = upper - lower + 1
    return (n**2 - 1) / (6 * n)


def lmoment3(lower, upper):
    shape = pt.broadcast_arrays(lower, upper)[0]
    return pt.zeros_like(shape)


def lmoment4(lower, upper):
    n = upper - lower + 1
    return -1.0 / (n**2)


def entropy(lower, upper):
    return discrete_entropy(lower, upper + 1, logpdf, lower, upper)


def pdf(x, lower, upper):
    return pt.exp(logpdf(x, lower, upper))


def cdf(x, lower, upper):
    prob = (pt.floor(x) - lower + 1) / (upper - lower + 1)
    return cdf_bounds(prob, x, lower, upper)


def ppf(q, lower, upper):
    n = upper - lower + 1
    x_vals = lower + pt.ceil(q * n) - 1
    return ppf_bounds_disc(x_vals, q, lower, upper)


def sf(x, lower, upper):
    return 1.0 - cdf(x, lower, upper)


def isf(q, lower, upper):
    return ppf(1.0 - q, lower, upper)


def rvs(lower, upper, size=None, random_state=None):
    return pt.random.integers(lower, upper + 1, size=size, rng=random_state)


def logpdf(x, lower, upper):
    n = upper - lower + 1
    return pt.switch(pt.and_(pt.ge(x, lower), pt.le(x, upper)), -pt.log(n), -pt.inf)


def logcdf(x, lower, upper):
    return pt.log(cdf(x, lower, upper))


def logsf(x, lower, upper):
    return pt.log1p(-cdf(x, lower, upper))
