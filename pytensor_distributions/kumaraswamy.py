import pytensor.tensor as pt
from pytensor.tensor.special import betaln
from pytensor.tensor.xlogx import xlogy0

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_cont, sf_bounds


def mean(a, b):
    return b * pt.exp(pt.gammaln(1 + 1 / a) + pt.gammaln(b) - pt.gammaln(1 + 1 / a + b))


def mode(a, b):
    a_b, b_b = pt.broadcast_arrays(a, b)
    condition = (a_b > 1) | (b_b > 1)
    mode_val = ((a_b - 1) / (a_b * b_b - 1)) ** (1 / a_b)
    return pt.where(condition, mode_val, 0.5)


def median(a, b):
    return ppf(0.5, a, b)


def var(a, b):
    m_1 = mean(a, b)
    m_2 = b * pt.exp(pt.gammaln(1 + 2 / a) + pt.gammaln(b) - pt.gammaln(1 + 2 / a + b))
    return m_2 - m_1**2


def std(a, b):
    return pt.sqrt(var(a, b))


def skewness(a, b):
    m_1 = mean(a, b)
    variance = var(a, b)
    m_3 = b * pt.exp(pt.gammaln(1 + 3 / a) + pt.gammaln(b) - pt.gammaln(1 + 3 / a + b))
    return (m_3 - 3 * m_1 * variance - m_1**3) / variance**1.5


def kurtosis(a, b):
    m_1 = mean(a, b)
    variance = var(a, b)
    m_2 = b * pt.exp(pt.gammaln(1 + 2 / a) + pt.gammaln(b) - pt.gammaln(1 + 2 / a + b))
    m_3 = b * pt.exp(pt.gammaln(1 + 3 / a) + pt.gammaln(b) - pt.gammaln(1 + 3 / a + b))
    m_4 = b * pt.exp(pt.gammaln(1 + 4 / a) + pt.gammaln(b) - pt.gammaln(1 + 4 / a + b))
    return (m_4 + m_1 * (-4 * m_3 + m_1 * (6 * m_2 - 3 * m_1**2))) / variance**2 - 3


def lmoment1(a, b):
    return mean(a, b)


def lmoment2(a, b):
    inv_a = 1.0 / a
    term1 = b * pt.exp(betaln(1.0 + inv_a, b))
    term2 = 2.0 * b * pt.exp(betaln(1.0 + inv_a, 2.0 * b))
    return term1 - term2


def lmoment3(a, b):
    inv_a = 1.0 / a
    l2 = lmoment2(a, b)
    term1 = b * pt.exp(betaln(1.0 + inv_a, b))
    term2 = 6.0 * b * pt.exp(betaln(1.0 + inv_a, 2.0 * b))
    term3 = 6.0 * b * pt.exp(betaln(1.0 + inv_a, 3.0 * b))
    return (term1 - term2 + term3) / l2


def lmoment4(a, b):
    inv_a = 1.0 / a
    l2 = lmoment2(a, b)
    term1 = b * pt.exp(betaln(1.0 + inv_a, b))
    term2 = 12.0 * b * pt.exp(betaln(1.0 + inv_a, 2.0 * b))
    term3 = 30.0 * b * pt.exp(betaln(1.0 + inv_a, 3.0 * b))
    term4 = 20.0 * b * pt.exp(betaln(1.0 + inv_a, 4.0 * b))
    return (term1 - term2 + term3 - term4) / l2


def entropy(a, b):
    h_b = pt.psi(b + 1) + pt.euler_gamma
    return (1 - 1 / b) + (1 - 1 / a) * h_b - pt.log(a) - pt.log(b)


def cdf(x, a, b):
    prob = 1 - (1 - x**a) ** b
    return cdf_bounds(prob, x, 0, 1)


def pdf(x, a, b):
    return pt.exp(logpdf(x, a, b))


def ppf(q, a, b):
    x_val = (1 - (1 - q) ** (1 / b)) ** (1 / a)
    return ppf_bounds_cont(x_val, q, 0, 1)


def sf(x, a, b):
    return sf_bounds((1 - x**a) ** b, x, 0, 1)


def isf(x, a, b):
    return ppf(1 - x, a, b)


def rvs(a, b, size=None, random_state=None):
    u = pt.random.uniform(0, 1, size=size, rng=random_state)
    return ppf(u, a, b)


def logpdf(x, a, b):
    return pt.switch(
        pt.bitwise_or(pt.le(x, 0), pt.ge(x, 1)),
        -pt.inf,
        pt.log(a * b) + xlogy0(a - 1, x) + xlogy0(b - 1, 1 - x**a),
    )


def logsf(x, a, b):
    return pt.switch(pt.le(x, 0), 0.0, pt.switch(pt.ge(x, 1), -pt.inf, b * pt.log(1 - x**a)))


def logcdf(x, a, b):
    return pt.switch(pt.le(x, 0), -pt.inf, pt.switch(pt.ge(x, 1), 0.0, pt.log(1 - (1 - x**a) ** b)))
