import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(nu, tau2):
    return pt.switch(pt.gt(nu, 2), nu * tau2 / (nu - 2), pt.inf)


def mode(nu, tau2):
    return nu * tau2 / (nu + 2)


def median(nu, tau2):
    return ppf(0.5, nu, tau2)


def var(nu, tau2):
    return pt.switch(
        pt.gt(nu, 4),
        2 * nu**2 * tau2**2 / ((nu - 2) ** 2 * (nu - 4)),
        pt.inf,
    )


def std(nu, tau2):
    return pt.sqrt(var(nu, tau2))


def skewness(nu, tau2):
    return pt.switch(pt.gt(nu, 6), 4 / (nu - 6) * pt.sqrt(2 * (nu - 4)), pt.nan)


def kurtosis(nu, tau2):
    return pt.switch(pt.gt(nu, 8), (12 * (5 * nu - 22)) / ((nu - 6) * (nu - 8)), pt.nan)


def lmoment1(nu, tau2):
    return mean(nu, tau2)


def lmoment2(nu, tau2):
    return pt.switch(pt.gt(nu, 4), _lmoments(ppf, nu, tau2, r=2), pt.inf)


def lmoment3(nu, tau2):
    return pt.switch(pt.gt(nu, 6), _lmoments(ppf, nu, tau2, r=3), pt.nan)


def lmoment4(nu, tau2):
    return pt.switch(pt.gt(nu, 8), _lmoments(ppf, nu, tau2, r=4), pt.nan)


def entropy(nu, tau2):
    h_nu = nu / 2
    return h_nu + pt.log(h_nu * tau2) + pt.gammaln(h_nu) - (1 + h_nu) * pt.digamma(h_nu)


def cdf(x, nu, tau2):
    h_nu = nu / 2
    return cdf_bounds(pt.gammaincc(h_nu, h_nu * tau2 / x), x, 0, pt.inf)


def isf(x, nu, tau2):
    return ppf(1 - x, nu, tau2)


def pdf(x, nu, tau2):
    return pt.exp(logpdf(x, nu, tau2))


def ppf(q, nu, tau2):
    h_nu = nu / 2
    vals = h_nu * tau2 / pt.gammaincinv(h_nu, 1 - q)
    return ppf_bounds_cont(vals, q, 0, pt.inf)


def sf(x, nu, tau2):
    return pt.exp(logsf(x, nu, tau2))


def rvs(nu, tau2, size=None, random_state=None):
    return (nu * tau2) / pt.random.chisquare(nu, rng=random_state, size=size)


def logcdf(x, nu, tau2):
    h_nu = nu / 2
    return pt.switch(pt.le(x, 0), -pt.inf, pt.log(pt.gammaincc(h_nu, h_nu * tau2 / x)))


def logpdf(x, nu, tau2):
    h_nu = nu / 2
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        -(pt.log(x) * (h_nu + 1))
        - (h_nu * tau2) / x
        + pt.log(tau2) * h_nu
        - pt.gammaln(h_nu)
        + pt.log(h_nu) * h_nu,
    )


def logsf(x, nu, tau2):
    h_nu = nu / 2
    return pt.switch(pt.le(x, 0), 0.0, pt.log(pt.gammainc(h_nu, h_nu * tau2 / x)))
