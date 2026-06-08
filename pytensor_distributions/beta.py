import pytensor.tensor as pt
from pytensor.tensor.math import betaincinv
from pytensor.tensor.special import betaln, xlogy

from pytensor_distributions.helper import ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(alpha, beta):
    return alpha / (alpha + beta)


def mode(alpha, beta):
    alpha_b, beta_b = pt.broadcast_arrays(alpha, beta)
    result = pt.full_like(alpha_b, pt.nan)

    result = pt.where(pt.equal(alpha_b, 1) & (beta_b > 1), 0.0, result)
    result = pt.where(pt.equal(beta_b, 1) & (alpha_b > 1), 1.0, result)
    result = pt.where((alpha_b > 1) & (beta_b > 1), (alpha_b - 1) / (alpha_b + beta_b - 2), result)

    return result


def median(alpha, beta):
    return ppf(0.5, alpha, beta)


def var(alpha, beta):
    return (alpha * beta) / (pt.pow(alpha + beta, 2) * (alpha + beta + 1))


def std(alpha, beta):
    return pt.sqrt(var(alpha, beta))


def skewness(alpha, beta):
    psc = alpha + beta
    result = pt.where(
        pt.eq(alpha, beta),
        0.0,
        (2 * (beta - alpha) * pt.sqrt(psc + 1)) / ((psc + 2) * pt.sqrt(alpha * beta)),
    )
    return result


def kurtosis(alpha, beta):
    psc = alpha + beta
    prod = alpha * beta
    result = (
        6
        * (pt.abs(alpha - beta) ** 2 * (psc + 1) - prod * (psc + 2))
        / (prod * (psc + 2) * (psc + 3))
    )
    return result


def lmoment1(alpha, beta):
    return mean(alpha, beta)


def lmoment2(alpha, beta):
    return _lmoments(ppf, alpha, beta, r=2)


def lmoment3(alpha, beta):
    return _lmoments(ppf, alpha, beta, r=3)


def lmoment4(alpha, beta):
    return _lmoments(ppf, alpha, beta, r=4)


def entropy(alpha, beta):
    return (
        betaln(alpha, beta)
        - (alpha - 1) * pt.psi(alpha)
        - (beta - 1) * pt.psi(beta)
        + (alpha + beta - 2) * pt.psi(alpha + beta)
    )


def cdf(x, alpha, beta):
    return pt.exp(logcdf(x, alpha, beta))


def isf(x, alpha, beta):
    return ppf(1 - x, alpha, beta)


def pdf(x, alpha, beta):
    return pt.exp(logpdf(x, alpha, beta))


def ppf(q, alpha, beta):
    return ppf_bounds_cont(betaincinv(alpha, beta, q), q, 0.0, 1.0)


def sf(x, alpha, beta):
    return pt.exp(logsf(x, alpha, beta))


def rvs(alpha, beta, size=None, random_state=None):
    return pt.random.beta(alpha, beta, rng=random_state, size=size)


def logcdf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.switch(
            pt.lt(x, 1),
            pt.log(pt.betainc(alpha, beta, x)),
            0,
        ),
    )


def logpdf(x, alpha, beta):
    return pt.switch(
        pt.bitwise_or(pt.lt(x, 0), pt.gt(x, 1)),
        -pt.inf,
        (xlogy((alpha - 1), x) + xlogy((beta - 1), 1 - x))
        - (xlogy((alpha + beta - 1), 1) + betaln(alpha, beta)),
    )


def logsf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0),
        0,
        pt.switch(
            pt.lt(x, 1),
            pt.log(pt.betainc(beta, alpha, 1 - x)),
            -pt.inf,
        ),
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
