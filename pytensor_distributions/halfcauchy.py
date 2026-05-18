import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont


def mean(beta):
    return pt.full_like(beta, pt.inf)


def mode(beta):
    return pt.zeros_like(beta)


def median(beta):
    return ppf(0.5, beta)


def var(beta):
    return pt.full_like(beta, pt.inf)


def std(beta):
    return pt.full_like(beta, pt.inf)


def skewness(beta):
    return pt.full_like(beta, pt.nan)


def kurtosis(beta):
    return pt.full_like(beta, pt.nan)


def lmoment1(beta):
    return mean(beta)


def lmoment2(beta):
    return pt.full_like(beta, pt.inf)


def lmoment3(beta):
    return pt.full_like(beta, pt.inf)


def lmoment4(beta):
    return pt.full_like(beta, pt.inf)


def entropy(beta):
    return pt.log(2 * pt.pi * beta)


def cdf(x, beta):
    return pt.exp(logcdf(x, beta))


def logcdf(x, beta):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.log(2 * pt.arctan(x / beta) / pt.pi),
    )


def isf(x, beta):
    return ppf(1 - x, beta)


def pdf(x, beta):
    return pt.exp(logpdf(x, beta))


def ppf(q, beta):
    x_val = beta * pt.tan(pt.pi / 2 * q)
    return ppf_bounds_cont(x_val, q, 0, pt.inf)


def sf(x, beta):
    return 1 - cdf(x, beta)


def rvs(beta, size=None, random_state=None):
    uniform_samples = pt.random.uniform(0, 1, rng=random_state, size=size)
    return beta * pt.tan(pt.pi / 2 * uniform_samples)


def logpdf(x, beta):
    return pt.where(
        pt.lt(x, 0), -pt.inf, pt.log(2) - pt.log(pt.pi * beta) - pt.log(1 + (x / beta) ** 2)
    )


def logsf(x, beta):
    return pt.log1mexp(logcdf(x, beta))
