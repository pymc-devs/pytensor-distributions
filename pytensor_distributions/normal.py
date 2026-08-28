import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont


def mean(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, mu)


def mode(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, mu)


def median(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, mu)


def var(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, pt.square(sigma))


def std(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, sigma)


def skewness(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, 0.0)


def kurtosis(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, 0.0)


def lmoment1(mu, sigma):
    return mean(mu, sigma)


def lmoment2(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    # sigma * sqrt(pi)
    return pt.full_like(shape, sigma * 0.5641895835477563)


def lmoment3(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.zeros_like(shape)


def lmoment4(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    # \frac{30}{\pi} \arcsin(\frac{1}{2}) - \frac{9}{\sqrt{\pi}}$
    return pt.full_like(shape, 0.122601719540890947)


def entropy(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, 0.5 * (pt.log(2 * pt.pi * pt.e * sigma**2)))


def cdf(x, mu, sigma):
    return 0.5 * (1 + pt.erf((x - mu) / (sigma * 2**0.5)))


def isf(x, mu, sigma):
    return ppf(1 - x, mu, sigma)


def pdf(x, mu, sigma):
    return 1 / pt.sqrt(2 * pt.pi * sigma**2) * pt.exp(-0.5 * ((x - mu) / sigma) ** 2)


def ppf(q, mu, sigma):
    return ppf_bounds_cont(mu + sigma * 2**0.5 * pt.erfinv(2 * q - 1), q, -pt.inf, pt.inf)


def sf(x, mu, sigma):
    return pt.exp(logsf(x, mu, sigma))


def rvs(mu, sigma, size=None, random_state=None):
    return pt.random.normal(mu, sigma, rng=random_state, size=size, return_next_rng=True)[1]


def logcdf(x, mu, sigma):
    z = (x - mu) / sigma
    return pt.switch(
        pt.lt(z, -1.0),
        pt.log(pt.erfcx(-z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
        pt.log1p(-pt.erfc(z / pt.sqrt(2.0)) / 2.0),
    )


def logpdf(x, mu, sigma):
    return -0.5 * pt.pow((x - mu) / sigma, 2) - pt.log(pt.sqrt(2.0 * pt.pi)) - pt.log(sigma)


def logsf(x, mu, sigma):
    z = (x - mu) / sigma
    return pt.switch(
        pt.gt(z, 1.0),
        pt.log(pt.erfcx(z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
        pt.log1p(-pt.erfc(-z / pt.sqrt(2.0)) / 2.0),
    )
