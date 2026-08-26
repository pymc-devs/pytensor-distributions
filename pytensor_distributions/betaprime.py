import pytensor.tensor as pt
from pytensor.tensor.math import betaincinv
from pytensor.tensor.special import betaln, xlogy

from pytensor_distributions.helper import ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(alpha, beta):
    return pt.switch(pt.gt(beta, 1), alpha / (beta - 1), pt.inf)


def mode(alpha, beta):
    alpha_b, beta_b = pt.broadcast_arrays(alpha, beta)
    return pt.where(alpha_b > 1, (alpha_b - 1) / (beta_b + 1), 0.0)


def median(alpha, beta):
    return ppf(0.5, alpha, beta)


def var(alpha, beta):
    return pt.switch(
        pt.gt(beta, 2),
        (alpha * (alpha + beta - 1)) / (pt.pow(beta - 1, 2) * (beta - 2)),
        pt.inf,
    )


def std(alpha, beta):
    return pt.sqrt(var(alpha, beta))


def skewness(alpha, beta):
    return pt.switch(
        pt.gt(beta, 3),
        (2 * (2 * alpha + beta - 1) / (beta - 3))
        * pt.sqrt((beta - 2) / (alpha * (alpha + beta - 1))),
        pt.nan,
    )


def kurtosis(alpha, beta):
    psc = alpha + beta - 1
    return pt.switch(
        pt.gt(beta, 4),
        6
        * (alpha * psc * (5 * beta - 11) + pt.pow(beta - 1, 2) * (beta - 2))
        / (alpha * psc * (beta - 3) * (beta - 4)),
        pt.nan,
    )


def lmoment1(alpha, beta):
    return mean(alpha, beta)


def lmoment2(alpha, beta):
    return pt.switch(pt.gt(beta, 1), _lmoments(ppf, alpha, beta, r=2), pt.inf)


def lmoment3(alpha, beta):
    return pt.switch(pt.gt(beta, 1), _lmoments(ppf, alpha, beta, r=3), pt.inf)


def lmoment4(alpha, beta):
    return pt.switch(pt.gt(beta, 1), _lmoments(ppf, alpha, beta, r=4), pt.inf)


def entropy(alpha, beta):
    return (
        betaln(alpha, beta)
        - (alpha - 1) * pt.psi(alpha)
        - (beta + 1) * pt.psi(beta)
        + (alpha + beta) * pt.psi(alpha + beta)
    )


def cdf(x, alpha, beta):
    return pt.exp(logcdf(x, alpha, beta))


def isf(x, alpha, beta):
    return ppf(1 - x, alpha, beta)


def pdf(x, alpha, beta):
    return pt.exp(logpdf(x, alpha, beta))


def ppf(q, alpha, beta):
    x = betaincinv(alpha, beta, q)
    return ppf_bounds_cont(x / (1 - x), q, 0.0, pt.inf)


def sf(x, alpha, beta):
    return pt.exp(logsf(x, alpha, beta))


def rvs(alpha, beta, size=None, random_state=None):
    rng, g1 = pt.random.gamma(
        shape=alpha, scale=1, size=size, rng=random_state, return_next_rng=True)
    g2 = pt.random.gamma(
        shape=beta, scale=1, size=size, rng=rng, return_next_rng=True)[1]
    return g1 / g2


def logpdf(x, alpha, beta):
    z = pt.switch(pt.or_(pt.isinf(x), pt.le(x, 0)), 1.0, x)
    result = (
        xlogy(alpha - 1, z)
        - (alpha + beta) * pt.log1p(z)
        - betaln(alpha, beta)
    )
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        pt.switch(pt.isinf(x), -pt.inf, result),
    )


def logcdf(x, alpha, beta):
    z = pt.switch(pt.isinf(x), 1.0, x)
    result = pt.log(pt.betainc(alpha, beta, z / (1 + z)))
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.switch(pt.isinf(x), 0.0, result),
    )


def logsf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0),
        0,
        pt.log(pt.betainc(beta, alpha, 1 / (1 + x))),
    )


def from_mu_nu(mu, nu):
    alpha = mu * (1 + nu)
    beta = 2 + nu
    return alpha, beta


def to_mu_nu(alpha, beta):
    mu = alpha / (beta - 1)
    nu = beta - 2
    return mu, nu
