import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont


def mean(alpha, m):
    return pt.switch(pt.gt(alpha, 1), alpha * m / (alpha - 1), pt.inf)


def mode(alpha, m):
    shape = pt.broadcast_arrays(alpha, m)[0]
    return pt.full_like(shape, m)


def median(alpha, m):
    return m * 2 ** (1 / alpha)


def var(alpha, m):
    return pt.switch(pt.gt(alpha, 2), m**2 * alpha / ((alpha - 1) ** 2 * (alpha - 2)), pt.inf)


def std(alpha, m):
    return pt.sqrt(var(alpha, m))


def skewness(alpha, m):
    alpha = pt.broadcast_arrays(alpha, m)[0]
    return pt.switch(
        pt.gt(alpha, 3), 2 * (1 + alpha) / (alpha - 3) * pt.sqrt(1 - 2 / alpha), pt.nan
    )


def kurtosis(alpha, m):
    alpha = pt.broadcast_arrays(alpha, m)[0]
    return pt.switch(
        pt.gt(alpha, 4),
        6 * (alpha**3 + alpha**2 - 6 * alpha - 2) / (alpha * (alpha - 3) * (alpha - 4)),
        pt.nan,
    )


def lmoment1(alpha, m):
    return mean(alpha, m)


def lmoment2(alpha, m):
    return pt.switch(pt.gt(alpha, 1), (alpha * m) / ((alpha - 1) * (2 * alpha - 1)), pt.inf)


def lmoment3(alpha, m):
    return pt.switch(pt.gt(alpha, 1), (1 + alpha) / (3 * alpha - 1), pt.inf)


def lmoment4(alpha, m):
    return pt.switch(
        pt.gt(alpha, 1),
        (alpha + 1.0) * (2.0 * alpha + 1.0) / ((3.0 * alpha - 1.0) * (4.0 * alpha - 1.0)),
        pt.inf,
    )


def entropy(alpha, m):
    return pt.log(m / alpha) + (1 + 1 / alpha)


def cdf(x, alpha, m):
    return pt.switch(pt.lt(x, m), pt.zeros_like(x), 1 - (m / x) ** alpha)


def pdf(x, alpha, m):
    return pt.exp(logpdf(x, alpha, m))


def ppf(q, alpha, m):
    x_val = m * (1 - q) ** (-1 / alpha)
    return ppf_bounds_cont(x_val, q, m, pt.inf)


def sf(x, alpha, m):
    return 1 - cdf(x, alpha, m)


def isf(x, alpha, m):
    return ppf(1 - x, alpha, m)


def rvs(alpha, m, size=None, random_state=None):
    u = pt.random.uniform(size=size, rng=random_state)
    return m / (1 - u) ** (1 / alpha)


def logcdf(x, alpha, m):
    return pt.switch(pt.lt(x, m), -pt.inf, pt.log(1 - (m / x) ** alpha))


def logpdf(x, alpha, m):
    return pt.switch(
        pt.lt(x, m), -pt.inf, pt.log(alpha) + alpha * pt.log(m) - (alpha + 1) * pt.log(x)
    )


def logsf(x, alpha, m):
    return pt.log(sf(x, alpha, m))
