import pytensor.tensor as pt

from pytensor_distributions.helper import (
    cdf_bounds,
    discrete_entropy,
    discrete_kurtosis,
    discrete_mean,
    discrete_skewness,
    discrete_variance,
    ppf_bounds_disc,
)


def mean(q, beta):
    return discrete_mean(ppf, pdf, q, beta)


def mode(q, beta):
    return pt.cast(pt.floor(-pt.log(-pt.log(q)) / pt.log(beta)), "int64")


def median(q, beta):
    return ppf(0.5, q, beta)


def var(q, beta):
    return discrete_variance(ppf, pdf, q, beta)


def std(q, beta):
    return pt.sqrt(var(q, beta))


def skewness(q, beta):
    return discrete_skewness(ppf, pdf, q, beta)


def kurtosis(q, beta):
    return discrete_kurtosis(ppf, pdf, q, beta)


def entropy(q, beta):
    # discrete Weibull can have very heavy tails, so we limit the upper bound
    # we may want to find a better way to handle this
    upper = pt.min([ppf(0.9999, q, beta), 1e4])
    return discrete_entropy(0, upper, logpdf, q, beta)


def cdf(x, q, beta):
    return cdf_bounds(1 - q ** ((x + 1) ** beta), x, 0, pt.inf)


def isf(x, q, beta):
    return ppf(1 - x, q, beta)


def pdf(x, q, beta):
    return pt.switch(
        pt.lt(x, 0),
        0,
        q ** (pt.floor(x) ** beta) - q ** ((pt.floor(x) + 1) ** beta),
    )


def ppf(p, q, beta):
    x_val = pt.ceil((pt.log(1 - p) / pt.log(q)) ** (1 / beta) - 1)
    return ppf_bounds_disc(x_val, p, 0, pt.inf)


def sf(x, q, beta):
    return pt.switch(
        pt.lt(x, 0),
        1,
        q ** ((pt.floor(x) + 1) ** beta),
    )


def rvs(q, beta, size=None, random_state=None):
    return ppf(
        pt.random.uniform(0, 1, rng=random_state, size=size, return_next_rng=True)[1], q, beta
    )


def logcdf(x, q, beta):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.log(1 - q ** ((pt.floor(x) + 1) ** beta)),
    )


def logpdf(x, q, beta):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.log(q ** (pt.floor(x) ** beta) - q ** ((pt.floor(x) + 1) ** beta)),
    )


def logsf(x, q, beta):
    return pt.switch(
        pt.lt(x, 0),
        0,
        pt.log(q ** ((pt.floor(x) + 1) ** beta)),
    )
