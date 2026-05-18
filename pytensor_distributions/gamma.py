import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(alpha, beta):
    return alpha / beta


def mode(alpha, beta):
    return pt.switch(pt.le(alpha, 1), 0.0, (alpha - 1) / beta)


def median(alpha, beta):
    return ppf(0.5, alpha, beta)


def var(alpha, beta):
    return alpha / pt.square(beta)


def std(alpha, beta):
    return pt.sqrt(alpha) / beta


def skewness(alpha, beta):
    shape = pt.broadcast_arrays(alpha, beta)[0]
    return pt.full_like(shape, 2 / pt.sqrt(alpha))


def kurtosis(alpha, beta):
    shape = pt.broadcast_arrays(alpha, beta)[0]
    return pt.full_like(shape, 6 / alpha)


def lmoment1(alpha, beta):
    return mean(alpha, beta)


def lmoment2(alpha, beta):
    return _lmoments(ppf, alpha, beta, r=2)


def lmoment3(alpha, beta):
    return _lmoments(ppf, alpha, beta, r=3)


def lmoment4(alpha, beta):
    return _lmoments(ppf, alpha, beta, r=4)


def entropy(alpha, beta):
    return alpha - pt.log(beta) + pt.gammaln(alpha) + (1 - alpha) * pt.digamma(alpha)


def cdf(x, alpha, beta):
    return pt.exp(logcdf(x, alpha, beta))


def pdf(x, alpha, beta):
    return pt.exp(logpdf(x, alpha, beta))


def ppf(q, alpha, beta):
    x_val = pt.gammaincinv(alpha, q) / beta
    return ppf_bounds_cont(x_val, q, 0.0, pt.inf)


def sf(x, alpha, beta):
    return 1 - cdf(x, alpha, beta)


def isf(x, alpha, beta):
    return ppf(1 - x, alpha, beta)


def rvs(alpha, beta, size=None, random_state=None):
    return pt.random.gamma(shape=alpha, scale=1 / beta, rng=random_state, size=size)


def logcdf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0.0),
        -pt.inf,
        pt.switch(
            pt.lt(x * beta, alpha),
            pt.log(pt.gammainc(alpha, x * beta)),
            pt.log1p(-pt.gammaincc(alpha, x * beta)),
        ),
    )


def logpdf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        (alpha - 1.0) * pt.log(x) - x * beta - pt.gammaln(alpha) + alpha * pt.log(beta),
    )


def logsf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0.0),
        0.0,
        pt.switch(
            pt.lt(x * beta, alpha),
            pt.log1p(-pt.gammainc(alpha, x * beta)),
            pt.log(pt.gammaincc(alpha, x * beta)),
        ),
    )


def from_mu_sigma(mu, sigma):
    alpha = mu**2 / sigma**2
    beta = mu / sigma**2
    return alpha, beta


def to_mu_sigma(alpha, beta):
    mu = alpha / beta
    sigma = alpha**0.5 / beta
    return mu, sigma
