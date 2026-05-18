import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont


def mean(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, mu)


def mode(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, mu)


def median(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, mu)


def var(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, s**2 * pt.pi**2 / 3)


def std(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, s * pt.pi / pt.sqrt(3))


def skewness(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, 0.0)


def kurtosis(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, 1.2)


def lmoment1(mu, s):
    return mean(mu, s)


def lmoment2(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, s)


def lmoment3(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.zeros_like(shape)


def lmoment4(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, 1 / 6)


def entropy(mu, s):
    shape = pt.broadcast_arrays(mu, s)[0]
    return pt.full_like(shape, pt.log(s) + 2)


def cdf(x, mu, s):
    return 1 / (1 + pt.exp(-(x - mu) / s))


def isf(x, mu, s):
    return ppf(1 - x, mu, s)


def pdf(x, mu, s):
    return pt.exp(logpdf(x, mu, s))


def ppf(q, mu, s):
    return ppf_bounds_cont(mu + s * pt.log(q / (1 - q)), q, -pt.inf, pt.inf)


def sf(x, mu, s):
    return pt.exp(logsf(x, mu, s))


def rvs(mu, s, size=None, random_state=None):
    return pt.random.logistic(mu, s, rng=random_state, size=size)


def logcdf(x, mu, s):
    return -pt.log1pexp(-(x - mu) / s)


def logpdf(x, mu, s):
    z = (x - mu) / s
    return pt.switch(pt.eq(x, -pt.inf), -pt.inf, -pt.log(s) - z - 2.0 * pt.log1p(pt.exp(-z)))


def logsf(x, mu, s):
    return -pt.log1pexp((x - mu) / s)
