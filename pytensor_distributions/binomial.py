import pytensor.tensor as pt
from pytensor.tensor.xlogx import xlogy0

from pytensor_distributions.helper import cdf_bounds, discrete_entropy
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf_discrete


def mean(n, p):
    return n * p


def mode(n, p):
    return pt.floor((n + 1) * p)


def median(n, p):
    return ppf(0.5, n, p)


def var(n, p):
    return n * p * (1 - p)


def std(n, p):
    return pt.sqrt(var(n, p))


def skewness(n, p):
    q = 1 - p
    return (q - p) / pt.sqrt(n * p * q)


def kurtosis(n, p):
    return (1 - 6 * p * (1 - p)) / (n * p * (1 - p))


def lmoment1(n, p):
    return mean(n, p)


def lmoment2(n, p):
    return _lmoments(ppf, n, p, r=2)


def lmoment3(n, p):
    return _lmoments(ppf, n, p, r=3)


def lmoment4(n, p):
    return _lmoments(ppf, n, p, r=4)


def entropy(n, p):
    return discrete_entropy(0, n + 1, logpdf, n, p)


def pdf(x, n, p):
    return pt.exp(logpdf(x, n, p))


def cdf(x, n, p):
    return cdf_bounds(pt.betainc(n - x, x + 1, 1 - p), x, 0, n)


def ppf(q, n, p):
    params = (n, p)
    return find_ppf_discrete(q, mean(n, p), 0, n, cdf, pdf, *params)


def sf(x, n, p):
    return 1.0 - cdf(x, n, p)


def isf(q, n, p):
    return ppf(1.0 - q, n, p)


def rvs(n, p, size=None, random_state=None):
    return pt.random.binomial(n, p, size=size, rng=random_state)


def logpdf(x, n, p):
    return pt.switch(
        pt.or_(pt.lt(x, 0), pt.gt(x, n)),
        -pt.inf,
        pt.gammaln(n + 1)
        - pt.gammaln(x + 1)
        - pt.gammaln(n - x + 1)
        + xlogy0(x, p)
        + (n - x) * pt.log1p(-p),
    )


def logcdf(x, n, p):
    return pt.log(cdf(x, n, p))


def logsf(x, n, p):
    return pt.log1p(-cdf(x, n, p))
