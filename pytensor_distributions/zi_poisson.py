import pytensor.tensor as pt

from pytensor_distributions import poisson as Poisson
from pytensor_distributions.helper import cdf_bounds, discrete_entropy, sf_bounds, zi_mode
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf_discrete


def mean(psi, mu):
    return psi * mu


def mode(psi, mu):
    base_mode = Poisson.mode(mu)
    return zi_mode(base_mode, logpdf, psi, mu)


def median(psi, mu):
    return ppf(0.5, psi, mu)


def var(psi, mu):
    return psi * (mu + (1 - psi) * pt.power(mu, 2))


def std(psi, mu):
    return pt.sqrt(var(psi, mu))


def skewness(psi, mu):
    ex1 = psi * mu
    ex2 = psi * (mu + pt.power(mu, 2))
    ex3 = psi * (mu + 3 * pt.power(mu, 2) + pt.power(mu, 3))

    mu_val = ex1
    mu2 = ex2 - pt.power(mu_val, 2)
    mu3 = ex3 - 3 * mu_val * ex2 + 2 * pt.power(mu_val, 3)

    return mu3 / pt.power(mu2, 1.5)


def kurtosis(psi, mu):
    ex1 = psi * mu
    ex2 = psi * (mu + pt.power(mu, 2))
    ex3 = psi * (mu + 3 * pt.power(mu, 2) + pt.power(mu, 3))
    ex4 = psi * (mu + 7 * pt.power(mu, 2) + 6 * pt.power(mu, 3) + pt.power(mu, 4))

    mu_val = ex1
    mu2 = ex2 - pt.power(mu_val, 2)
    mu4 = ex4 - 4 * mu_val * ex3 + 6 * pt.power(mu_val, 2) * ex2 - 3 * pt.power(mu_val, 4)

    return mu4 / pt.power(mu2, 2) - 3


def lmoment1(psi, mu):
    return mean(psi, mu)


def lmoment2(psi, mu):
    return _lmoments(ppf, psi, mu, r=2)


def lmoment3(psi, mu):
    return _lmoments(ppf, psi, mu, r=3)


def lmoment4(psi, mu):
    return _lmoments(ppf, psi, mu, r=4)


def entropy(psi, mu):
    return pt.switch(
        pt.lt(mu, 10),
        discrete_entropy(0, 25, logpdf, psi, mu),
        discrete_entropy(0, pt.cast(mu * 3, "int64"), logpdf, psi, mu),
    )


def pdf(x, psi, mu):
    x = pt.as_tensor_variable(x)
    result = pt.exp(logpdf(x, psi, mu))
    return pt.switch(
        pt.isinf(x) & pt.gt(x, 0),
        pt.nan,
        result,
    )


def logpdf(x, psi, mu):
    x = pt.as_tensor_variable(x)

    log_zero_prob = pt.log((1 - psi) + psi * pt.exp(-mu))

    base_logpdf = Poisson.logpdf(x, mu)
    log_nonzero_prob = pt.log(psi) + base_logpdf

    result = pt.switch(
        pt.or_(pt.lt(x, 0), pt.isinf(x)),
        -pt.inf,
        pt.switch(pt.eq(x, 0), log_zero_prob, log_nonzero_prob),
    )

    return pt.switch(
        pt.isinf(x) & pt.gt(x, 0),
        pt.nan,
        result,
    )


def cdf(x, psi, mu):
    base_cdf = Poisson.cdf(x, mu)
    zi_cdf = (1 - psi) + psi * base_cdf
    return cdf_bounds(zi_cdf, x, 0, pt.inf)


def ppf(q, psi, mu):
    params = (psi, mu)
    return find_ppf_discrete(q, mean(1, mu), 0, pt.inf, cdf, pdf, *params)


def sf(x, psi, mu):
    base_sf = Poisson.sf(x, mu)
    zi_sf = psi * base_sf
    return sf_bounds(zi_sf, x, 0, pt.inf)


def isf(q, psi, mu):
    return ppf(1.0 - q, psi, mu)


def rvs(psi, mu, size=None, random_state=None):
    base_samples = pt.random.poisson(mu, size=size, rng=random_state)
    mask = pt.random.bernoulli(psi, size=size)
    return pt.cast(mask, "int64") * base_samples


def logcdf(x, psi, mu):
    base_cdf = Poisson.cdf(x, mu)
    result = pt.log1p(psi * (base_cdf - 1))
    return pt.switch(
        pt.or_(pt.lt(x, 0), pt.isinf(x)),
        pt.switch(pt.lt(x, 0), -pt.inf, 0.0),
        result,
    )


def logsf(x, psi, mu):
    return pt.log(sf(x, psi, mu))


def expect(x, psi, mu, func):
    if func is None:
        return x * pdf(x, psi, mu)
    return func(x) * pdf(x, psi, mu)
