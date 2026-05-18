import pytensor.tensor as pt
from pytensor.tensor.math import betaincinv
from pytensor.tensor.special import betaln
from pytensor.tensor.xlogx import xlogy0

from pytensor_distributions.helper import ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(alpha, beta, lower, upper):
    return (alpha * upper + beta * lower) / (alpha + beta)


def mode(alpha, beta, lower, upper):
    alpha_b, beta_b, lower_b, upper_b = pt.broadcast_arrays(alpha, beta, lower, upper)
    result = pt.full_like(alpha_b, pt.nan)

    result = pt.where((alpha_b < 1) & (beta_b < 1), pt.nan, result)
    result = pt.where((alpha_b <= 1) & (beta_b > 1), lower_b, result)
    result = pt.where((beta_b <= 1) & (alpha_b > 1), upper_b, result)
    result = pt.where(
        (alpha_b > 1) & (beta_b > 1),
        lower_b + (alpha_b - 1) / (alpha_b + beta_b - 2) * (upper_b - lower_b),
        result,
    )
    return result


def median(alpha, beta, lower, upper):
    return ppf(0.5, alpha, beta, lower, upper)


def var(alpha, beta, lower, upper):
    return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)) * (upper - lower) ** 2


def std(alpha, beta, lower, upper):
    return pt.sqrt(var(alpha, beta, lower, upper))


def skewness(alpha, beta, lower, upper):
    psc = alpha + beta
    result = pt.where(
        pt.eq(alpha, beta),
        0.0,
        (2 * (beta - alpha) * pt.sqrt(psc + 1)) / ((psc + 2) * pt.sqrt(alpha * beta)),
    )
    return result


def kurtosis(alpha, beta, lower, upper):
    psc = alpha + beta
    prod = alpha * beta
    result = (
        6
        * (pt.abs(alpha - beta) ** 2 * (psc + 1) - prod * (psc + 2))
        / (prod * (psc + 2) * (psc + 3))
    )
    return result


def lmoment1(alpha, beta, lower, upper):
    return mean(alpha, beta, lower, upper)


def lmoment2(alpha, beta, lower, upper):
    return _lmoments(ppf, alpha, beta, lower, upper, r=2)


def lmoment3(alpha, beta, lower, upper):
    return _lmoments(ppf, alpha, beta, lower, upper, r=3)


def lmoment4(alpha, beta, lower, upper):
    return _lmoments(ppf, alpha, beta, lower, upper, r=4)


def entropy(alpha, beta, lower, upper):
    return (
        betaln(alpha, beta)
        - (alpha - 1) * pt.psi(alpha)
        - (beta - 1) * pt.psi(beta)
        + (alpha + beta - 2) * pt.psi(alpha + beta)
        + pt.log(upper - lower)
    )


def cdf(x, alpha, beta, lower, upper):
    return pt.exp(logcdf(x, alpha, beta, lower, upper))


def isf(x, alpha, beta, lower, upper):
    return ppf(1 - x, alpha, beta, lower, upper)


def pdf(x, alpha, beta, lower, upper):
    return pt.exp(logpdf(x, alpha, beta, lower, upper))


def ppf(q, alpha, beta, lower, upper):
    x_val = betaincinv(alpha, beta, q) * (upper - lower) + lower
    return ppf_bounds_cont(x_val, q, lower, upper)


def sf(x, alpha, beta, lower, upper):
    return 1 - cdf(x, alpha, beta, lower, upper)


def rvs(alpha, beta, lower, upper, size=None, random_state=None):
    beta_samples = pt.random.beta(alpha, beta, rng=random_state, size=size)
    return beta_samples * (upper - lower) + lower


def logcdf(x, alpha, beta, lower, upper):
    x_normalized = (x - lower) / (upper - lower)
    return pt.switch(
        pt.lt(x, lower),
        -pt.inf,
        pt.switch(pt.gt(x, upper), 0.0, pt.log(pt.betainc(alpha, beta, x_normalized))),
    )


def logpdf(x, alpha, beta, lower, upper):
    return pt.switch(
        pt.bitwise_or(pt.lt(x, lower), pt.gt(x, upper)),
        -pt.inf,
        (xlogy0((alpha - 1), (x - lower)) + xlogy0((beta - 1), (upper - x)))
        - (xlogy0((alpha + beta - 1), (upper - lower)) + betaln(alpha, beta)),
    )


def logsf(x, alpha, beta, lower, upper):
    x_normalized = (x - lower) / (upper - lower)
    return pt.switch(
        pt.lt(x, lower),
        0.0,
        pt.switch(pt.gt(x, upper), -pt.inf, pt.log(pt.betainc(beta, alpha, 1 - x_normalized))),
    )


def from_mu_sigma(mu, sigma):
    nu = mu * (1 - mu) / sigma**2 - 1
    alpha = mu * nu
    beta = (1 - mu) * nu
    return alpha, beta


def from_mu_nu(mu, nu):
    alpha = mu * nu
    beta = (1 - mu) * nu
    return alpha, beta


def to_mu_sigma(alpha, beta):
    alpha_plus_beta = alpha + beta
    mu = alpha / alpha_plus_beta
    sigma = (alpha * beta) ** 0.5 / alpha_plus_beta / (alpha_plus_beta + 1) ** 0.5
    return mu, sigma
