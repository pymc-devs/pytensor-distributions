import pytensor.tensor as pt
from pytensor.tensor.special import xlogy

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(nu):
    return nu


def mode(nu):
    return pt.maximum(nu - 2, 0)


def median(nu):
    return ppf(0.5, nu)


def var(nu):
    return nu * 2


def std(nu):
    return pt.sqrt(var(nu))


def skewness(nu):
    return pt.sqrt(8 / nu)


def kurtosis(nu):
    return 12 / nu


def lmoment1(nu):
    return mean(nu)


def lmoment2(nu):
    return _lmoments(ppf, nu, r=2)


def lmoment3(nu):
    return _lmoments(ppf, nu, r=3)


def lmoment4(nu):
    return _lmoments(ppf, nu, r=4)


def entropy(nu):
    h_nu = nu / 2
    return h_nu + pt.log(2) + pt.gammaln(h_nu) + (1 - h_nu) * pt.digamma(h_nu)


def cdf(x, nu):
    return cdf_bounds(pt.gammainc(nu / 2, x / 2), x, 0, pt.inf)


def ppf(q, nu):
    vals = 2 * pt.gammaincinv(nu / 2, q)
    return ppf_bounds_cont(vals, q, 0, pt.inf)


def pdf(x, nu):
    return pt.exp(logpdf(x, nu))


def rvs(nu, size=None, random_state=None):
    return pt.random.chisquare(nu, rng=random_state, size=size)


def logcdf(x, nu):
    return pt.switch(
        pt.lt(x, 0.0),
        -pt.inf,
        pt.switch(
            pt.lt(x, nu),
            pt.log(pt.gammainc(nu / 2, x / 2)),
            pt.log1p(-pt.gammaincc(nu / 2, x / 2)),
        ),
    )


def logpdf(x, nu):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        xlogy(nu / 2 - 1, x) - x / 2 - pt.gammaln(nu / 2) - (nu * pt.log(2)) / 2,
    )


def sf(x, nu):
    return 1 - cdf(x, nu)


def isf(x, nu):
    return ppf(1 - x, nu)


def logsf(x, nu):
    return pt.switch(
        pt.lt(x, 0.0),
        0.0,
        pt.switch(
            pt.lt(x, nu),
            pt.log1p(-pt.gammainc(nu / 2, x / 2)),
            pt.log(pt.gammaincc(nu / 2, x / 2)),
        ),
    )
