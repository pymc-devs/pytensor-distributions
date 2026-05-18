import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_cont


def mean(mu, b):
    shape = pt.broadcast_arrays(mu, b)[0]
    return pt.full_like(shape, mu)


def mode(mu, b):
    shape = pt.broadcast_arrays(mu, b)[0]
    return pt.full_like(shape, mu)


def median(mu, b):
    shape = pt.broadcast_arrays(mu, b)[0]
    return pt.full_like(shape, mu)


def var(mu, b):
    _, b_b = pt.broadcast_arrays(mu, b)
    return 2.0 * pt.square(b_b)


def std(mu, b):
    _, b_b = pt.broadcast_arrays(mu, b)
    return pt.sqrt(2) * b_b


def skewness(mu, b):
    shape = pt.broadcast_arrays(mu, b)[0]
    return pt.full_like(shape, 0.0)


def kurtosis(mu, b):
    shape = pt.broadcast_arrays(mu, b)[0]
    return pt.full_like(shape, 3.0)


def lmoment1(mu, b):
    return mean(mu, b)


def lmoment2(mu, b):
    shape = pt.broadcast_arrays(mu, b)[0]
    return pt.full_like(shape, b * 0.75)


def lmoment3(mu, b):
    shape = pt.broadcast_arrays(mu, b)[0]
    return pt.zeros_like(shape)


def lmoment4(mu, b):
    shape = pt.broadcast_arrays(mu, b)[0]
    return pt.full_like(shape, 0.2361111111111111)


def entropy(mu, b):
    _, b_b = pt.broadcast_arrays(mu, b)
    return pt.log(2.0 * b_b) + 1.0


def pdf(x, mu, b):
    return 0.5 / b * pt.exp(-pt.abs(x - mu) / b)


def cdf(x, mu, b):
    result = pt.switch(pt.lt(x, mu), 0.5 * pt.exp((x - mu) / b), 1.0 - 0.5 * pt.exp(-(x - mu) / b))
    return cdf_bounds(result, x, -pt.inf, pt.inf)


def sf(x, mu, b):
    return 1.0 - cdf(x, mu, b)


def ppf(q, mu, b):
    result = pt.switch(pt.lt(q, 0.5), mu + b * pt.log(2.0 * q), mu - b * pt.log(2.0 * (1.0 - q)))
    return ppf_bounds_cont(result, q, -pt.inf, pt.inf)


def isf(q, mu, b):
    return ppf(1.0 - q, mu, b)


def rvs(mu, b, size=None, random_state=None):
    return pt.random.laplace(mu, b, rng=random_state, size=size)


def logpdf(x, mu, b):
    return pt.log(0.5) - pt.abs((x - mu) / b) - pt.log(b)


def logcdf(x, mu, b):
    y = (x - mu) / b
    return pt.switch(
        pt.le(y, 0),
        pt.log(0.5) + y,
        pt.switch(
            pt.gt(y, 1),
            pt.log1p(-0.5 * pt.exp(-y)),
            pt.log(1 - 0.5 * pt.exp(-y)),
        ),
    )


def logsf(x, mu, b):
    return logcdf(-x, -mu, b)
