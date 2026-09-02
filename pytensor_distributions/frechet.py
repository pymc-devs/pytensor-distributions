import pytensor.tensor as pt
from pytensor.tensor.special import gamma

from pytensor_distributions import weibull
from pytensor_distributions.helper import ppf_bounds_cont


def mean(alpha, sigma):
    return pt.switch(
        pt.gt(alpha, 1),
        sigma * gamma(1 - 1 / alpha),
        pt.switch(pt.eq(alpha, 1), pt.inf, pt.nan),
    )


def var(alpha, sigma):
    return pt.switch(
        pt.gt(alpha, 2),
        sigma**2 * (gamma(1 - 2 / alpha) - gamma(1 - 1 / alpha) ** 2),
        pt.switch(pt.ge(alpha, 1), pt.inf, pt.nan),
    )


def std(alpha, sigma):
    return pt.sqrt(var(alpha, sigma))


def skewness(alpha, sigma):
    g1 = gamma(1 - 1 / alpha)
    g2 = gamma(1 - 2 / alpha)
    g3 = gamma(1 - 3 / alpha)
    sigma2 = sigma**2 * (g2 - g1**2)
    mu3 = sigma**3 * (g3 - 3 * g1 * g2 + 2 * g1**3)
    return pt.switch(alpha > 3, mu3 / sigma2 ** (3 / 2), pt.nan)


def kurtosis(alpha, sigma):
    g1 = gamma(1 - 1 / alpha)
    g2 = gamma(1 - 2 / alpha)
    g3 = gamma(1 - 3 / alpha)
    g4 = gamma(1 - 4 / alpha)
    sigma2 = sigma**2 * (g2 - g1**2)
    mu4 = sigma**4 * (g4 - 4 * g1 * g3 + 6 * g1**2 * g2 - 3 * g1**4)
    return pt.switch(alpha > 4, mu4 / sigma2**2 - 3, pt.nan)


def mode(alpha, sigma):
    return sigma * (alpha / (alpha + 1)) ** (1 / alpha)


def median(alpha, sigma):
    return sigma * pt.log(2) ** (-1 / alpha)


def entropy(alpha, sigma):
    return 1 + pt.euler_gamma * (1 + 1 / alpha) + pt.log(sigma / alpha)


def lmoment1(alpha, sigma):
    return mean(alpha, sigma)


def lmoment2(alpha, sigma):
    return pt.switch(
        alpha > 1,
        sigma * gamma(1 - 1 / alpha) * (2 ** (1 / alpha) - 1),
        pt.inf,
    )


def lmoment3(alpha, sigma):
    tau3 = (2 * 3 ** (1 / alpha) - 3 * 2 ** (1 / alpha) + 1) / (2 ** (1 / alpha) - 1)
    return pt.switch(alpha > 2, tau3, pt.inf)


def lmoment4(alpha, sigma):
    tau4 = (5 * 4 ** (1 / alpha) - 10 * 3 ** (1 / alpha) + 6 * 2 ** (1 / alpha) - 1) / (
        2 ** (1 / alpha) - 1
    )
    return pt.switch(alpha > 3, tau4, pt.inf)


def pdf(y, alpha, sigma):
    return pt.switch(
        pt.or_(pt.le(y, 0), pt.eq(y, pt.inf)),
        0.0,
        weibull.pdf(1 / y, alpha, 1 / sigma) / y**2,
    )


def logpdf(y, alpha, sigma):
    return pt.switch(
        pt.or_(pt.le(y, 0), pt.eq(y, pt.inf)),
        -pt.inf,
        weibull.logpdf(1 / y, alpha, 1 / sigma) - 2 * pt.log(y),
    )


def cdf(y, alpha, sigma):
    return pt.switch(
        pt.le(y, 0),
        0.0,
        pt.switch(pt.eq(y, pt.inf), 1.0, weibull.sf(1 / y, alpha, 1 / sigma)),
    )


def logcdf(y, alpha, sigma):
    return pt.switch(
        pt.le(y, 0),
        -pt.inf,
        pt.switch(pt.eq(y, pt.inf), 0.0, weibull.logsf(1 / y, alpha, 1 / sigma)),
    )


def sf(y, alpha, sigma):
    return pt.switch(
        pt.le(y, 0),
        1.0,
        pt.switch(pt.eq(y, pt.inf), 0.0, weibull.cdf(1 / y, alpha, 1 / sigma)),
    )


def logsf(y, alpha, sigma):
    return pt.switch(
        pt.le(y, 0),
        0.0,
        pt.switch(pt.eq(y, pt.inf), -pt.inf, weibull.logcdf(1 / y, alpha, 1 / sigma)),
    )


def ppf(q, alpha, sigma):
    x_val = 1 / weibull.isf(q, alpha, 1 / sigma)
    return ppf_bounds_cont(x_val, q, 0.0, pt.inf)


def isf(q, alpha, sigma):
    x_val = 1 / weibull.ppf(q, alpha, 1 / sigma)
    return pt.switch(
        pt.or_(pt.lt(q, 0), pt.gt(q, 1)),
        pt.nan,
        pt.switch(pt.eq(q, 0), pt.inf, pt.switch(pt.eq(q, 1), 0.0, x_val)),
    )


def rvs(alpha, sigma, size=None, random_state=None):
    return 1 / weibull.rvs(alpha, 1 / sigma, size=size, random_state=random_state)
