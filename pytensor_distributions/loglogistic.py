import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(alpha, beta):
    b = pt.pi / beta
    result = alpha * b / pt.sin(b)
    return pt.switch(pt.gt(beta, 1), result, pt.nan)


def mode(alpha, beta):
    return pt.switch(
        pt.gt(beta, 1),
        alpha * ((beta - 1) / (beta + 1)) ** (1 / beta),
        pt.zeros_like(alpha * beta),
    )


def median(alpha, beta):
    shape = pt.broadcast_arrays(alpha, beta)[0]
    return pt.full_like(shape, alpha)


def var(alpha, beta):
    b = pt.pi / beta
    result = alpha**2 * (2 * b / pt.sin(2 * b) - b**2 / pt.sin(b) ** 2)
    return pt.switch(pt.gt(beta, 2), result, pt.nan)


def std(alpha, beta):
    return pt.sqrt(var(alpha, beta))


def skewness(alpha, beta):
    b = pt.pi / beta
    m1 = b / pt.sin(b)
    m2 = 2 * b / pt.sin(2 * b)
    m3 = 3 * b / pt.sin(3 * b)
    mu = alpha * m1
    sigma2 = alpha**2 * (m2 - m1**2)
    sigma = pt.sqrt(sigma2)
    result = (alpha**3 * m3 - 3 * mu * sigma2 - mu**3) / sigma**3
    return pt.switch(pt.gt(beta, 3), result, pt.nan)


def kurtosis(alpha, beta):
    b = pt.pi / beta
    m1 = b / pt.sin(b)
    m2 = 2 * b / pt.sin(2 * b)
    m3 = 3 * b / pt.sin(3 * b)
    m4 = 4 * b / pt.sin(4 * b)
    mu = alpha * m1
    sigma2 = alpha**2 * (m2 - m1**2)
    sigma = pt.sqrt(sigma2)
    skew = (alpha**3 * m3 - 3 * mu * sigma2 - mu**3) / sigma**3
    m_s = mu / sigma
    result = (alpha / sigma) ** 4 * m4 - 4 * skew * m_s - 6 * m_s**2 - m_s**4 - 3
    return pt.switch(pt.gt(beta, 4), result, pt.nan)


def lmoment1(alpha, beta):
    return mean(alpha, beta)


def lmoment2(alpha, beta):
    return pt.switch(pt.gt(beta, 1), _lmoments(ppf, alpha, beta, r=2, n_points=100), pt.inf)


def lmoment3(alpha, beta):
    return pt.switch(pt.gt(beta, 1), _lmoments(ppf, alpha, beta, r=3, n_points=100), pt.inf)


def lmoment4(alpha, beta):
    return pt.switch(pt.gt(beta, 1), _lmoments(ppf, alpha, beta, r=4, n_points=100), pt.inf)


def entropy(alpha, beta):
    return pt.log(alpha) - pt.log(beta) + 2


def pdf(x, alpha, beta):
    return pt.exp(logpdf(x, alpha, beta))


def logpdf(x, alpha, beta):
    z = x / alpha
    result = pt.log(beta) - pt.log(alpha) + (beta - 1) * pt.log(z) - 2 * pt.log1p(z**beta)
    return pt.switch(pt.le(x, 0), -pt.inf, pt.switch(pt.isinf(x), -pt.inf, result))


def cdf(x, alpha, beta):
    z = x / alpha
    prob = 1 / (1 + z ** (-beta))
    return cdf_bounds(prob, x, 0, pt.inf)


def logcdf(x, alpha, beta):
    z = x / alpha
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        -pt.log1p(z ** (-beta)),
    )


def sf(x, alpha, beta):
    z = x / alpha
    return pt.switch(
        pt.le(x, 0),
        1.0,
        1 / (1 + z**beta),
    )


def logsf(x, alpha, beta):
    z = x / alpha
    return pt.switch(
        pt.le(x, 0),
        0.0,
        -pt.log1p(z**beta),
    )


def ppf(q, alpha, beta):
    x_val = alpha * (q / (1 - q)) ** (1 / beta)
    return ppf_bounds_cont(x_val, q, 0, pt.inf)


def isf(q, alpha, beta):
    return ppf(1 - q, alpha, beta)


def rvs(alpha, beta, size=None, random_state=None):
    u = pt.random.uniform(size=size, rng=random_state, return_next_rng=True)[1]
    return alpha * (u / (1 - u)) ** (1 / beta)
