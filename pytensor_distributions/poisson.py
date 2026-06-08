import pytensor.tensor as pt
from pytensor.tensor.special import xlogy

from pytensor_distributions.helper import cdf_bounds, discrete_entropy, sf_bounds
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf_discrete


def mean(mu):
    return mu


def mode(mu):
    return pt.floor(mu)


def median(mu):
    return pt.floor(mu)


def var(mu):
    return pt.as_tensor_variable(mu)


def std(mu):
    return pt.sqrt(mu)


def skewness(mu):
    return 1.0 / pt.sqrt(mu)


def kurtosis(mu):
    return 1.0 / mu


def lmoment1(mu):
    return mean(mu)


def lmoment2(mu):
    return _lmoments(ppf, mu, r=2)


def lmoment3(mu):
    return _lmoments(ppf, mu, r=3)


def lmoment4(mu):
    return _lmoments(ppf, mu, r=4)


def entropy(mu):
    # Use explicit sum for small mu and Stirling's approximation for large mu
    return pt.switch(
        pt.lt(mu, 10),
        discrete_entropy(0, 25, logpdf, mu),
        0.5 * pt.log(2.0 * pt.pi * pt.e * mu) - 1.0 / (12.0 * mu),
    )


def pdf(x, mu):
    return pt.exp(logpdf(x, mu))


def cdf(x, mu):
    return cdf_bounds(pt.gammaincc(x + 1, mu), x, 0, pt.inf)


def ppf(q, mu):
    params = (mu,)
    return find_ppf_discrete(q, mu, 0, pt.inf, cdf, pdf, *params)


def sf(x, mu):
    return sf_bounds(pt.gammainc(pt.floor(x) + 1, mu), x, 0, pt.inf)


def isf(q, mu):
    return ppf(1.0 - q, mu)


def rvs(mu, size=None, random_state=None):
    return pt.random.poisson(mu, rng=random_state, size=size, return_next_rng=True)[1]


def logpdf(x, mu):
    return xlogy(x, mu) - pt.gammaln(x + 1) - mu


def logcdf(x, mu):
    return pt.log(cdf(x, mu))


def logsf(x, mu):
    return pt.log(sf(x, mu))


def expect(x, mu, func):
    if func is None:
        return x * pdf(x, mu)
    return func(x) * pdf(x, mu)
