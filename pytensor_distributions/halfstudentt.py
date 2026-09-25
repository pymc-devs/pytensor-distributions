import pytensor.tensor as pt
from pytensor.tensor.math import betaincinv
from pytensor.tensor.special import betaln

from pytensor_distributions.halfnormal import cdf as halfnormal_cdf
from pytensor_distributions.halfnormal import entropy as halfnormal_entropy
from pytensor_distributions.halfnormal import isf as halfnormal_isf
from pytensor_distributions.halfnormal import logcdf as halfnormal_logcdf
from pytensor_distributions.halfnormal import logpdf as halfnormal_logpdf
from pytensor_distributions.halfnormal import logsf as halfnormal_logsf
from pytensor_distributions.halfnormal import sf as halfnormal_sf
from pytensor_distributions.helper import LOG2, cdf_bounds, isf_bounds_cont, ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(nu, sigma):
    gamma0 = pt.exp(pt.gammaln((nu + 1) / 2))
    gamma1 = pt.exp(pt.gammaln(nu / 2))
    gamma_nonfinite = pt.isinf(gamma0) | pt.isinf(gamma1)

    mean_finite = 2 * sigma * pt.sqrt(nu / pt.pi) * (gamma0 / (gamma1 * (nu - 1)))

    mean_approx = sigma * pt.sqrt(2 / pt.pi)
    mean_val = pt.where(gamma_nonfinite, mean_approx, mean_finite)

    return pt.where(nu > 1, mean_val, pt.inf)


def mode(nu, sigma):
    _, sigma_b = pt.broadcast_arrays(nu, sigma)
    return pt.zeros_like(sigma_b)


def median(nu, sigma):
    return ppf(0.5, nu, sigma)


def var(nu, sigma):
    gamma0 = pt.exp(pt.gammaln((nu + 1) / 2))
    gamma1 = pt.exp(pt.gammaln(nu / 2))
    gamma_nonfinite = pt.isinf(gamma0) | pt.isinf(gamma1)

    var_finite = sigma**2 * (
        (nu / (nu - 2)) - ((4 * nu) / (pt.pi * (nu - 1) ** 2)) * (gamma0 / gamma1) ** 2
    )

    var_approx = sigma**2 * (1 - 2.0 / pt.pi)
    var_val = pt.where(gamma_nonfinite, var_approx, var_finite)

    return pt.where(nu > 2, var_val, pt.inf)


def std(nu, sigma):
    variance = var(nu, sigma)
    return pt.switch(pt.isinf(variance) | pt.isnan(variance), variance, pt.sqrt(variance))


def skewness(nu, sigma):
    nu_b, _ = pt.broadcast_arrays(nu, sigma)
    return pt.full_like(nu_b, pt.nan)


def kurtosis(nu, sigma):
    nu_b, _ = pt.broadcast_arrays(nu, sigma)
    return pt.full_like(nu_b, pt.nan)


def lmoment1(nu, sigma):
    return mean(nu, sigma)


def lmoment2(nu, sigma):
    return pt.switch(nu > 2, _lmoments(ppf, nu, sigma, r=2), pt.inf)


def lmoment3(nu, sigma):
    return pt.switch(nu > 3, _lmoments(ppf, nu, sigma, r=3), pt.inf)


def lmoment4(nu, sigma):
    return pt.switch(nu > 4, _lmoments(ppf, nu, sigma, r=4), pt.inf)


def entropy(nu, sigma):
    # we use a halfnormal approximation for large nu
    return pt.switch(
        pt.gt(nu, 1e5),
        halfnormal_entropy(sigma),
        pt.log(sigma)
        + 0.5 * (nu + 1) * (pt.psi(0.5 * (nu + 1)) - pt.psi(0.5 * nu))
        + pt.log(pt.sqrt(nu))
        + betaln(0.5 * nu, 0.5)
        - LOG2,
    )


def _factor(x, nu, sigma):
    x_norm = x / sigma
    return 0.5 * pt.betainc(0.5 * nu, 0.5, nu / (x_norm**2 + nu))


def cdf(x, nu, sigma):
    # we use a halfnormal approximation for large nu
    x_norm = x / sigma
    factor = _factor(x, nu, sigma)
    cdf_ = pt.switch(pt.lt(x_norm, 0), factor, 1 - factor) * 2 - 1
    halft_cdf = cdf_bounds(cdf_, x, 0, pt.inf)

    return pt.switch(pt.gt(nu, 1e5), halfnormal_cdf(x, sigma), halft_cdf)


def isf(x, nu, sigma):
    inv_factor = pt.sqrt(nu / betaincinv(0.5 * nu, 0.5, x) - nu)
    return pt.switch(
        pt.gt(nu, 1e5),
        halfnormal_isf(x, sigma),
        isf_bounds_cont(inv_factor * sigma, x, 0, pt.inf),
    )


def pdf(x, nu, sigma):
    return pt.exp(logpdf(x, nu, sigma))


def ppf(q, nu, sigma):
    q_factor = (q + 1) / 2
    inv_factor = pt.switch(
        pt.lt(q_factor, 0.5),
        betaincinv(0.5 * nu, 0.5, 2 * q_factor),
        pt.sqrt(nu / betaincinv(0.5 * nu, 0.5, 2 - 2 * q_factor) - nu),
    )
    return ppf_bounds_cont(inv_factor * sigma, q, 0, pt.inf)


def sf(x, nu, sigma):
    x_norm = x / sigma
    factor = _factor(x, nu, sigma)
    halft_sf = pt.switch(pt.lt(x_norm, 0), 1.0, 2 * factor)
    return pt.switch(pt.gt(nu, 1e5), halfnormal_sf(x, sigma), halft_sf)


def rvs(nu, sigma, size=None, random_state=None):
    t_samples = pt.random.t(nu, rng=random_state, size=size, return_next_rng=True)[1]
    return pt.abs(t_samples * sigma)


def logcdf(x, nu, sigma):
    x_norm = x / sigma
    factor = _factor(x, nu, sigma)
    halft_logcdf = pt.switch(pt.lt(x_norm, 0), -pt.inf, pt.log1p(-2 * factor))
    return pt.switch(pt.gt(nu, 1e5), halfnormal_logcdf(x, sigma), halft_logcdf)


def logpdf(x, nu, sigma):
    # we use a halfnormal approximation for large nu
    halft_logpdf = pt.where(
        pt.lt(x, 0),
        -pt.inf,
        (
            pt.gammaln((nu + 1) / 2)
            - pt.gammaln(nu / 2)
            - 0.5 * pt.log(nu * pt.pi * sigma**2)
            - 0.5 * (nu + 1) * pt.log(1 + (x / sigma) ** 2 / nu)
            + LOG2
        ),
    )

    return pt.switch(pt.gt(nu, 1e5), halfnormal_logpdf(x, sigma), halft_logpdf)


def logsf(x, nu, sigma):
    x_norm = x / sigma
    factor = _factor(x, nu, sigma)
    halft_logsf = pt.switch(pt.lt(x_norm, 0), 0.0, LOG2 + pt.log(factor))
    return pt.switch(pt.gt(nu, 1e5), halfnormal_logsf(x, sigma), halft_logsf)
