import pytensor.tensor as pt
from pytensor.tensor.special import gamma, xlogy

from pytensor_distributions.helper import ppf_bounds_cont


def mean(alpha, beta):
    return beta * gamma(1 + 1 / alpha)


def mode(alpha, beta):
    return pt.switch(alpha > 1, beta * ((alpha - 1) / alpha) ** (1 / alpha), 0)


def median(alpha, beta):
    return beta * pt.log(2) ** (1 / alpha)


def var(alpha, beta):
    return beta**2 * gamma(1 + 2 / alpha) - mean(alpha, beta) ** 2


def std(alpha, beta):
    return pt.sqrt(var(alpha, beta))


def skewness(alpha, beta):
    mu = mean(alpha, beta)
    sigma = std(alpha, beta)
    m_s = mu / sigma
    return gamma(1 + 3 / alpha) * (beta / sigma) ** 3 - 3 * m_s - m_s**3


def kurtosis(alpha, beta):
    mu = mean(alpha, beta)
    sigma = std(alpha, beta)
    skew = skewness(alpha, beta)
    m_s = mu / sigma
    return (beta / sigma) ** 4 * gamma(1 + 4 / alpha) - 4 * skew * m_s - 6 * m_s**2 - m_s**4 - 3


def lmoment1(alpha, beta):
    return mean(alpha, beta)


def lmoment2(alpha, beta):
    g = pt.gamma(1.0 + 1.0 / alpha)
    return beta * (1.0 - 2.0 ** (-1.0 / alpha)) * g


def lmoment3(alpha, beta):
    inv_a = 1.0 / alpha
    c1 = 1.0 - 2.0 ** (-1.0 - inv_a)
    c2 = 1.0 - 2.0 * 2.0 ** (-1.0 - inv_a) + 3.0 ** (-1.0 - inv_a)
    return (6.0 * c2 - 6.0 * c1 + 1.0) / (2.0 * c1 - 1.0)


def lmoment4(alpha, beta):
    inv_a = 1.0 / alpha
    denom = 1.0 - 2.0 ** (-inv_a)
    num = 1.0 - 6.0 * (2.0 ** (-inv_a)) + 10.0 * (3.0 ** (-inv_a)) - 5.0 * (4.0 ** (-inv_a))
    return num / denom


def entropy(alpha, beta):
    return pt.euler_gamma * (1 - 1 / alpha) + pt.log(beta / alpha) + 1


def cdf(x, alpha, beta):
    return pt.exp(logcdf(x, alpha, beta))


def isf(x, alpha, beta):
    return ppf(1 - x, alpha, beta)


def pdf(x, alpha, beta):
    return pt.exp(logpdf(x, alpha, beta))


def ppf(q, alpha, beta):
    x_val = beta * (-pt.log(1 - q)) ** (1 / alpha)
    return ppf_bounds_cont(x_val, q, 0.0, pt.inf)


def sf(x, alpha, beta):
    return pt.exp(logsf(x, alpha, beta))


def rvs(alpha, beta, size=None, random_state=None):
    return pt.random.weibull(alpha, rng=random_state, size=size, return_next_rng=True)[1] * beta


def logcdf(x, alpha, beta):
    a = (x / beta) ** alpha

    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.log1mexp(-a),
    )


def logpdf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.log(alpha / beta) + xlogy(alpha - 1, x / beta) - (x / beta) ** alpha,
    )


def logsf(x, alpha, beta):
    return pt.switch(pt.lt(x, 0), 0, -((x / beta) ** alpha))
