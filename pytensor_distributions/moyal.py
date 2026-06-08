import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def mean(mu, sigma):
    return mu + sigma * (pt.euler_gamma + pt.log(2))


def mode(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, mu)


def median(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, mu + sigma * 0.7875976)


def var(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, pt.square(sigma) * (pt.pi**2) / 2)


def std(mu, sigma):
    return pt.sqrt(var(mu, sigma))


def skewness(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, 1.5351416)


def kurtosis(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, 4)


def lmoment1(mu, sigma):
    return mean(mu, sigma)


def lmoment2(mu, sigma):
    return _lmoments(ppf, mu, sigma, r=2)


def lmoment3(mu, sigma):
    return _lmoments(ppf, mu, sigma, r=3)


def lmoment4(mu, sigma):
    return _lmoments(ppf, mu, sigma, r=4)


def entropy(mu, sigma):
    return pt.log(sigma) + 2.0541199


def cdf(x, mu, sigma):
    z_val = (x - mu) / sigma
    return 1 - pt.erf(pt.exp(-z_val / 2) * (2**-0.5))


def pdf(x, mu, sigma):
    return pt.exp(logpdf(x, mu, sigma))


def ppf(q, mu, sigma):
    x_val = sigma * -pt.log(2.0 * pt.erfinv(1 - q) ** 2) + mu
    return ppf_bounds_cont(x_val, q, -pt.inf, pt.inf)


def sf(x, mu, sigma):
    return 1 - cdf(x, mu, sigma)


def isf(x, mu, sigma):
    return ppf(1 - x, mu, sigma)


def rvs(mu, sigma, size=None, random_state=None):
    u = pt.random.uniform(size=size, rng=random_state, return_next_rng=True)[1]
    return ppf(u, mu, sigma)


def logcdf(x, mu, sigma):
    z_val = (x - mu) / sigma
    return pt.log(1 - pt.erf(pt.exp(-z_val / 2) * (2**-0.5)))


def logpdf(x, mu, sigma):
    z_val = (x - mu) / sigma
    return -(1 / 2) * (z_val + pt.exp(-z_val)) - pt.log(sigma) - (1 / 2) * pt.log(2 * pt.pi)


def logsf(x, mu, sigma):
    return pt.log(sf(x, mu, sigma))
