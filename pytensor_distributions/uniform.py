import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont


def mean(lower, upper):
    return (upper + lower) / 2


def mode(lower, upper):
    shape = pt.broadcast_arrays(lower, upper)[0]
    return pt.full_like(shape, pt.nan)


def median(lower, upper):
    return mean(lower, upper)


def var(lower, upper):
    return (upper - lower) ** 2 / 12


def std(lower, upper):
    return pt.sqrt(var(lower, upper))


def skewness(lower, upper):
    shape = pt.broadcast_arrays(lower, upper)[0]
    return pt.zeros_like(shape)


def kurtosis(lower, upper):
    shape = pt.broadcast_arrays(lower, upper)[0]
    return pt.full_like(shape, -6 / 5)


def lmoment1(lower, upper):
    return mean(lower, upper)


def lmoment2(lower, upper):
    return (upper - lower) / 6


def lmoment3(lower, upper):
    shape = pt.broadcast_arrays(lower, upper)[0]
    return pt.zeros_like(shape)


def lmoment4(lower, upper):
    shape = pt.broadcast_arrays(lower, upper)[0]
    return pt.zeros_like(shape)


def entropy(lower, upper):
    return pt.log(upper - lower)


def cdf(x, lower, upper):
    return pt.switch(
        pt.lt(x, lower),
        pt.zeros_like(x),
        pt.switch(pt.gt(x, upper), pt.ones_like(x), (x - lower) / (upper - lower)),
    )


def pdf(x, lower, upper):
    return pt.exp(logpdf(x, lower, upper))


def ppf(q, lower, upper):
    x_vals = lower + q * (upper - lower)
    return ppf_bounds_cont(x_vals, q, lower, upper)


def sf(x, lower, upper):
    return 1 - cdf(x, lower, upper)


def isf(x, lower, upper):
    return ppf(1 - x, lower, upper)


def rvs(lower, upper, size=None, random_state=None):
    return pt.random.uniform(lower, upper, size=size, rng=random_state, return_next_rng=True)[1]


def logcdf(x, lower, upper):
    return pt.switch(
        pt.lt(x, lower),
        -pt.inf,
        pt.switch(pt.gt(x, upper), 0, pt.log((x - lower) / (upper - lower))),
    )


def logpdf(x, lower, upper):
    log_height = -pt.log(upper - lower)
    return pt.switch(pt.or_(pt.lt(x, lower), pt.gt(x, upper)), -pt.inf, log_height)


def logsf(x, lower, upper):
    return pt.log(sf(x, lower, upper))
