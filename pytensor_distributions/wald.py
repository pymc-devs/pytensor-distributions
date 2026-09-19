import pytensor.tensor as pt

from pytensor_distributions.helper import SQRT2, logdiffexp
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf


def mean(mu, lam):
    shape = pt.broadcast_arrays(mu, lam)[0]
    return pt.full_like(shape, mu)


def mode(mu, lam):
    ratio = 3 * mu / (2 * lam)
    return mu * (pt.sqrt(1 + ratio**2) - ratio)


def median(mu, lam):
    return ppf(0.5, mu, lam)


def var(mu, lam):
    return mu**3 / lam


def std(mu, lam):
    return pt.sqrt(var(mu, lam))


def skewness(mu, lam):
    return 3.0 * pt.sqrt(mu / lam)


def kurtosis(mu, lam):
    return 15.0 * mu / lam


def lmoment1(mu, lam):
    return mean(mu, lam)


def lmoment2(mu, lam):
    return _lmoments(ppf, mu, lam, r=2)


def lmoment3(mu, lam):
    return _lmoments(ppf, mu, lam, r=3)


def lmoment4(mu, lam):
    return _lmoments(ppf, mu, lam, r=4)


def entropy(mu, lam):
    x = 2 * lam / mu
    gamma_term = -pt.gammaincc(1e-8, x) * pt.gamma(1e-8)
    return 0.5 * pt.log((2 * pt.pi * pt.e * mu**3) / lam) + 1.5 * pt.exp(x) * gamma_term


def _wald_gini_terms(x, mu, lam):
    eps = 1e-12
    u = pt.sqrt(lam / (x + eps))
    v = x / mu
    sqrt2 = SQRT2
    return u * (v - 1) / sqrt2, u * (v + 1) / sqrt2


def cdf(x, mu, lam):
    return pt.exp(logcdf(x, mu, lam))


def _log_half_erfc(w):
    return pt.switch(
        pt.ge(w, 0),
        pt.log(0.5 * pt.erfcx(w)) - pt.square(w),
        pt.log1mexp(pt.log(0.5 * pt.erfcx(-w)) - pt.square(w)),
    )


def logcdf(x, mu, lam):
    w1, w2 = _wald_gini_terms(x, mu, lam)
    log_z1 = _log_half_erfc(-w1)
    log_z2 = 2 * lam / mu + _log_half_erfc(w2)
    result = pt.maximum(log_z1, log_z2) + pt.log1pexp(
        pt.minimum(log_z1, log_z2) - pt.maximum(log_z1, log_z2)
    )
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        pt.switch(pt.eq(x, pt.inf), 0.0, result),
    )


def isf(x, mu, lam):
    return ppf(1 - x, mu, lam)


def pdf(x, mu, lam):
    return pt.exp(logpdf(x, mu, lam))


def ppf(q, mu, lam):
    params = (mu, lam)
    return find_ppf(q, mode(mu, lam), 0, pt.inf, cdf, pdf, *params)


def sf(x, mu, lam):
    return pt.exp(logsf(x, mu, lam))


def rvs(mu, lam, size=None, random_state=None):
    return pt.random.wald(mu, lam, rng=random_state, size=size, return_next_rng=True)[1]


def logpdf(x, mu, lam):
    return pt.switch(
        pt.or_(pt.le(x, 0), pt.eq(x, pt.inf)),
        -pt.inf,
        0.5 * (pt.log(lam) - pt.log(2 * pt.pi) - 3 * pt.log(x) - lam * (x - mu) ** 2 / (mu**2 * x)),
    )


def logsf(x, mu, lam):
    w1, w2 = _wald_gini_terms(x, mu, lam)
    log_a = _log_half_erfc(w1)
    log_b = 2 * lam / mu + _log_half_erfc(w2)
    result = logdiffexp(log_a, log_b)
    result = pt.switch(pt.isnan(result), -pt.inf, result)
    return pt.switch(
        pt.le(x, 0),
        0.0,
        pt.switch(pt.eq(x, pt.inf), -pt.inf, result),
    )
