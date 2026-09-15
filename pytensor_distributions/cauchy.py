import pytensor.tensor as pt

from pytensor_distributions.helper import isf_bounds_cont, ppf_bounds_cont


def mean(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def mode(alpha, beta):
    return pt.broadcast_arrays(alpha, beta)[0]


def median(alpha, beta):
    return pt.broadcast_arrays(alpha, beta)[0]


def var(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def std(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def skewness(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def kurtosis(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def lmoment1(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def lmoment2(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def lmoment3(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def lmoment4(alpha, beta):
    alpha_b, _ = pt.broadcast_arrays(alpha, beta)
    return pt.full_like(alpha_b, pt.nan)


def entropy(alpha, beta):
    _, beta_b = pt.broadcast_arrays(alpha, beta)
    return pt.log(4 * pt.pi * beta_b)


def cdf(x, alpha, beta):
    return 1 / pt.pi * pt.arctan((x - alpha) / beta) + 0.5


def isf(x, alpha, beta):
    return isf_bounds_cont(alpha - beta * pt.tan(pt.pi * (x - 0.5)), x, -pt.inf, pt.inf)


def pdf(x, alpha, beta):
    return pt.exp(logpdf(x, alpha, beta))


def ppf(q, alpha, beta):
    return ppf_bounds_cont(alpha + beta * pt.tan(pt.pi * (q - 0.5)), q, -pt.inf, pt.inf)


def sf(x, alpha, beta):
    return pt.exp(logsf(x, alpha, beta))


def rvs(alpha, beta, size=None, random_state=None):
    return pt.random.cauchy(alpha, beta, rng=random_state, size=size, return_next_rng=True)[1]


def logcdf(x, alpha, beta):
    z = (x - alpha) / beta
    return pt.switch(
        pt.lt(z, 0),
        pt.log(pt.arctan2(1, -z) / pt.pi),
        pt.log1p(-pt.arctan2(1, z) / pt.pi),
    )


def logpdf(x, alpha, beta):
    return -pt.log(pt.pi) - pt.log(beta) - pt.log(1 + ((x - alpha) / beta) ** 2)


def logsf(x, alpha, beta):
    z = (x - alpha) / beta
    return pt.switch(
        pt.gt(z, 0),
        pt.log(pt.arctan2(1, z) / pt.pi),
        pt.log1p(-pt.arctan2(1, -z) / pt.pi),
    )
