import pytensor.tensor as pt
from pytensor.tensor.math import betaincinv
from pytensor.tensor.special import betaln

from pytensor_distributions.helper import ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.normal import cdf as normal_logcdf
from pytensor_distributions.normal import entropy as normal_entropy
from pytensor_distributions.normal import logcdf as normal_logpdf


def mean(nu, mu, sigma):
    nu_b, mu_b, _ = pt.broadcast_arrays(nu, mu, sigma)
    return pt.switch(nu_b > 1, mu_b, pt.nan)


def mode(nu, mu, sigma):
    shape = pt.broadcast_arrays(nu, mu, sigma)[0]
    return pt.full_like(shape, mu)


def median(nu, mu, sigma):
    shape = pt.broadcast_arrays(nu, mu, sigma)[0]
    return pt.full_like(shape, mu)


def var(nu, mu, sigma):
    nu_b, _, sigma_b = pt.broadcast_arrays(nu, mu, sigma)
    result = pt.full_like(nu_b, pt.nan)

    result = pt.switch(nu_b > 2, sigma_b**2 * nu_b / (nu_b - 2), result)
    result = pt.switch((nu_b > 1) & (nu_b <= 2), pt.inf, result)
    result = pt.switch(nu_b > 1e10, sigma_b**2, result)
    return result


def std(nu, mu, sigma):
    variance = var(nu, mu, sigma)
    return pt.switch(pt.isinf(variance) | pt.isnan(variance), variance, pt.sqrt(variance))


def skewness(nu, mu, sigma):
    nu_b = pt.broadcast_arrays(nu, mu, sigma)[0]
    return pt.switch(nu_b > 3, 0.0, pt.nan)


def kurtosis(nu, mu, sigma):
    nu_b = pt.broadcast_arrays(nu, mu, sigma)[0]
    result = pt.full_like(nu_b, pt.nan)

    result = pt.switch(nu_b > 4, 6 / (nu_b - 4), result)
    result = pt.switch((nu_b > 2) & (nu_b <= 4), pt.inf, result)
    return result


def lmoment1(nu, mu, sigma):
    return mean(nu, mu, sigma)


def lmoment2(nu, mu, sigma):
    return pt.switch(pt.gt(nu, 1), _lmoments(ppf, nu, mu, sigma, r=2), pt.inf)


def lmoment3(nu, mu, sigma):
    shape = pt.broadcast_arrays(nu, mu, sigma)[0]
    return pt.switch(pt.gt(nu, 1), pt.zeros_like(shape), pt.inf)


def lmoment4(nu, mu, sigma):
    return pt.switch(pt.gt(nu, 1), _lmoments(ppf, nu, mu, sigma, r=4), pt.inf)


def entropy(nu, mu, sigma):
    # we use a normal approximation for large nu
    return pt.switch(
        pt.gt(nu, 1e5),
        normal_entropy(mu, sigma),
        pt.log(sigma)
        + 0.5 * (nu + 1) * (pt.psi(0.5 * (nu + 1)) - pt.psi(0.5 * nu))
        + pt.log(pt.sqrt(nu))
        + betaln(0.5 * nu, 0.5),
    )


def cdf(x, nu, mu, sigma):
    return pt.exp(logcdf(x, nu, mu, sigma))


def isf(x, nu, mu, sigma):
    return ppf(1 - x, nu, mu, sigma)


def pdf(x, nu, mu, sigma):
    return pt.exp(logpdf(x, nu, mu, sigma))


def ppf(q, nu, mu, sigma):
    result = pt.switch(
        pt.abs(q - 0.5) < 1e-14,
        pt.zeros_like(q),
        pt.switch(
            pt.lt(q, 0.5),
            -pt.sqrt(nu) * pt.sqrt((1.0 / betaincinv(nu * 0.5, 0.5, 2.0 * q)) - 1.0),
            pt.sqrt(nu) * pt.sqrt((1.0 / betaincinv(nu * 0.5, 0.5, 2.0 * (1 - q))) - 1.0),
        ),
    )
    return ppf_bounds_cont(mu + result * sigma, q, -pt.inf, pt.inf)


def sf(x, nu, mu, sigma):
    return cdf(-x, nu, -mu, sigma)


def rvs(nu, mu, sigma, size=None, random_state=None):
    return pt.random.t(nu, mu, sigma, rng=random_state, size=size, return_next_rng=True)[1]


def logcdf(x, nu, mu, sigma):
    # we use a normal approximation for large nu
    z = (x - mu) / sigma
    factor = 0.5 * pt.betainc(0.5 * nu, 0.5, nu / (z**2 + nu))
    logcdf_t = pt.switch(pt.lt(z, 0), pt.log(factor), pt.log1p(-factor))
    return pt.switch(pt.gt(nu, 1e5), normal_logcdf(z, mu, sigma), logcdf_t)


def logpdf(x, nu, mu, sigma):
    # we use a normal approximation for large nu
    # scipy uses the pochhammer function, but the version available in pytensor
    # is not numerically stable for large nu
    return pt.switch(
        pt.gt(nu, 1e5),
        normal_logpdf(x, mu, sigma),
        pt.gammaln((nu + 1) / 2)
        - pt.gammaln(nu / 2)
        - 0.5 * pt.log(nu * pt.pi * sigma**2)
        - 0.5 * (nu + 1) * pt.log1p((x - mu) ** 2 / (sigma**2 * nu)),
    )


def logsf(x, nu, mu, sigma):
    return logcdf(-x, nu, -mu, sigma)
