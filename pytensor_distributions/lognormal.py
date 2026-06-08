import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(mu, sigma):
    return pt.exp(mu + sigma**2 / 2)


def mode(mu, sigma):
    return pt.exp(mu - sigma**2)


def median(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, pt.exp(mu))


def var(mu, sigma):
    return (pt.exp(sigma**2) - 1) * pt.exp(2 * mu + sigma**2)


def std(mu, sigma):
    return pt.sqrt(var(mu, sigma))


def skewness(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    exp_sigma2 = pt.exp(sigma**2)
    return pt.full_like(shape, (exp_sigma2 + 2) * pt.sqrt(exp_sigma2 - 1))


def kurtosis(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(
        shape, pt.exp(4 * sigma**2) + 2 * pt.exp(3 * sigma**2) + 3 * pt.exp(2 * sigma**2) - 6
    )


def lmoment1(mu, sigma):
    return mean(mu, sigma)


def lmoment2(mu, sigma):
    return pt.exp(mu + sigma**2 / 2) * pt.erf(sigma / 2)


def lmoment3(mu, sigma):
    return _lmoments(ppf, mu, sigma, r=3)


def lmoment4(mu, sigma):
    return _lmoments(ppf, mu, sigma, r=4)


def entropy(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, pt.log(pt.sqrt(2 * pt.pi) * sigma * pt.exp(mu + 0.5)))


def cdf(x, mu, sigma):
    prob = 0.5 * (1 + pt.erf((pt.log(x) - mu) / (sigma * pt.sqrt(2))))
    return cdf_bounds(prob, x, 0, pt.inf)


def isf(x, mu, sigma):
    return ppf(1 - x, mu, sigma)


def pdf(x, mu, sigma):
    return pt.exp(logpdf(x, mu, sigma))


def ppf(q, mu, sigma):
    return ppf_bounds_cont(pt.exp(mu + sigma * pt.sqrt(2) * pt.erfinv(2 * q - 1)), q, 0, pt.inf)


def sf(x, mu, sigma):
    return pt.exp(logsf(x, mu, sigma))


def rvs(mu, sigma, size=None, random_state=None):
    return pt.exp(pt.random.normal(mu, sigma, rng=random_state, size=size, return_next_rng=True)[1])


def logcdf(x, mu, sigma):
    z = (pt.log(x) - mu) / sigma
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        pt.switch(
            pt.lt(z, -1.0),
            pt.log(pt.erfcx(-z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
            pt.log1p(-pt.erfc(z / pt.sqrt(2.0)) / 2.0),
        ),
    )


def logpdf(x, mu, sigma):
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        -pt.log(x)
        - pt.log(sigma)
        - 0.5 * pt.log(2 * pt.pi)
        - 0.5 * ((pt.log(x) - mu) / sigma) ** 2,
    )


def logsf(x, mu, sigma):
    z = (pt.log(x) - mu) / sigma
    return pt.switch(
        pt.le(x, 0),
        0.0,  # sf(x) = 1 for x <= 0, so log(1) = 0
        pt.switch(
            pt.gt(z, 1.0),
            pt.log(pt.erfcx(z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
            pt.log1p(-0.5 * (1 + pt.erf(z / pt.sqrt(2.0)))),
        ),
    )
