import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(alpha, beta):
    return pt.switch(pt.gt(alpha, 1), beta / (alpha - 1), pt.inf)


def mode(alpha, beta):
    return beta / (alpha + 1)


def median(alpha, beta):
    return ppf(0.5, alpha, beta)


def var(alpha, beta):
    return pt.switch(pt.gt(alpha, 2), beta**2 / ((alpha - 1) ** 2 * (alpha - 2)), pt.inf)


def std(alpha, beta):
    return pt.sqrt(var(alpha, beta))


def skewness(alpha, beta):
    return pt.switch(pt.gt(alpha, 3), 4 * pt.sqrt(alpha - 2) / (alpha - 3), pt.nan)


def kurtosis(alpha, beta):
    return pt.switch(pt.gt(alpha, 4), 6 * (5 * alpha - 11) / ((alpha - 3) * (alpha - 4)), pt.nan)


def lmoment1(alpha, beta):
    return mean(alpha, beta)


def lmoment2(alpha, beta):
    return pt.switch(pt.gt(alpha, 1), _lmoments(ppf, alpha, beta, r=2), pt.inf)


def lmoment3(alpha, beta):
    return pt.switch(pt.gt(alpha, 1), _lmoments(ppf, alpha, beta, r=3), pt.inf)


def lmoment4(alpha, beta):
    return pt.switch(pt.gt(alpha, 1), _lmoments(ppf, alpha, beta, r=4), pt.inf)


def entropy(alpha, beta):
    h_regular = alpha - (alpha + 1.0) * pt.digamma(alpha) + pt.gammaln(alpha) + pt.log(beta)

    h_asymptotic = (
        (1 - 3 * pt.log(alpha) + pt.log(2) + pt.log(pt.pi)) / 2
        + 2 / 3 * alpha**-1
        + alpha**-2 / 12
        - alpha**-3 / 90
        - alpha**-4 / 120
        + pt.log(beta)
    )
    return pt.switch(pt.ge(alpha, 200), h_asymptotic, h_regular)


def cdf(x, alpha, beta):
    return cdf_bounds(pt.gammaincc(alpha, beta / x), x, 0, pt.inf)


def isf(x, alpha, beta):
    return ppf(1 - x, alpha, beta)


def pdf(x, alpha, beta):
    return pt.exp(logpdf(x, alpha, beta))


def ppf(q, alpha, beta):
    return ppf_bounds_cont(beta / pt.gammaincinv(alpha, 1 - q), q, 0, pt.inf)


def sf(x, alpha, beta):
    return pt.exp(logsf(x, alpha, beta))


def rvs(alpha, beta, size=None, random_state=None):
    return 1.0 / pt.random.gamma(alpha, beta, rng=random_state, size=size, return_next_rng=True)[1]


def logcdf(x, alpha, beta):
    return pt.switch(pt.le(x, 0), -pt.inf, pt.log(pt.gammaincc(alpha, beta / x)))


def logpdf(x, alpha, beta):
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        alpha * pt.log(beta) - pt.gammaln(alpha) - (alpha + 1) * pt.log(x) - beta / x,
    )


def logsf(x, alpha, beta):
    return pt.switch(pt.le(x, 0), 0.0, pt.log(pt.gammainc(alpha, beta / x)))


def from_mu_sigma(mu, sigma):
    """Convert mean and standard deviation to alpha and beta parameters."""
    alpha = mu**2 / sigma**2 + 2
    beta = mu**3 / sigma**2 + mu
    return alpha, beta


def to_mu(alpha, beta):
    return pt.switch(pt.gt(alpha, 1), beta / (alpha - 1), pt.nan)


def to_sigma(alpha, beta):
    return pt.switch(pt.gt(alpha, 2), beta / ((alpha - 1) * pt.sqrt(alpha - 2)), pt.nan)
