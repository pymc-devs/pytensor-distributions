import pytensor.tensor as pt

from pytensor_distributions import binomial as Binomial
from pytensor_distributions.helper import cdf_bounds, discrete_entropy, zi_mode
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf_discrete


def mean(psi, n, p):
    return psi * n * p


def mode(psi, n, p):
    base_mode = Binomial.mode(n, p)
    return zi_mode(base_mode, logpdf, psi, n, p)


def median(psi, n, p):
    return ppf(0.5, psi, n, p)


def var(psi, n, p):
    base_mean = n * p
    base_var = n * p * (1 - p)
    return psi * (base_var + (1 - psi) * pt.power(base_mean, 2))


def std(psi, n, p):
    return pt.sqrt(var(psi, n, p))


def skewness(psi, n, p):
    q = 1 - p
    base_ex1 = n * p
    base_ex2 = n * p * q + pt.power(base_ex1, 2)
    base_ex3 = n * p * q * (1 - 2 * p) + 3 * pt.power(n * p, 2) * q + pt.power(base_ex1, 3)

    ex1 = psi * base_ex1
    ex2 = psi * base_ex2
    ex3 = psi * base_ex3

    mu_val = ex1
    mu2 = ex2 - pt.power(mu_val, 2)
    mu3 = ex3 - 3 * mu_val * ex2 + 2 * pt.power(mu_val, 3)

    return mu3 / pt.power(mu2, 1.5)


def kurtosis(psi, n, p):
    ex1_fact = n * p
    ex2_fact = n * (n - 1) * pt.power(p, 2)
    ex3_fact = n * (n - 1) * (n - 2) * pt.power(p, 3)
    ex4_fact = n * (n - 1) * (n - 2) * (n - 3) * pt.power(p, 4)

    base_ex1 = ex1_fact
    base_ex2 = ex2_fact + ex1_fact
    base_ex3 = ex3_fact + 3 * ex2_fact + ex1_fact
    base_ex4 = ex4_fact + 6 * ex3_fact + 7 * ex2_fact + ex1_fact

    ex1 = psi * base_ex1
    ex2 = psi * base_ex2
    ex3 = psi * base_ex3
    ex4 = psi * base_ex4

    mu_val = ex1
    mu2 = ex2 - pt.power(mu_val, 2)
    mu4 = ex4 - 4 * mu_val * ex3 + 6 * pt.power(mu_val, 2) * ex2 - 3 * pt.power(mu_val, 4)

    return mu4 / pt.power(mu2, 2) - 3


def lmoment1(psi, n, p):
    return mean(psi, n, p)


def lmoment2(psi, n, p):
    return _lmoments(ppf, psi, n, p, r=2)


def lmoment3(psi, n, p):
    return _lmoments(ppf, psi, n, p, r=3)


def lmoment4(psi, n, p):
    return _lmoments(ppf, psi, n, p, r=4)


def entropy(psi, n, p):
    return discrete_entropy(0, n + 1, logpdf, psi, n, p)


def pdf(x, psi, n, p):
    return pt.exp(logpdf(x, psi, n, p))


def logpdf(x, psi, n, p):
    base_logpdf = Binomial.logpdf(x, n, p)

    log_zero_prob = pt.log((1 - psi) + psi * pt.power(1 - p, n))

    log_nonzero_prob = pt.log(psi) + base_logpdf

    return pt.switch(
        pt.or_(pt.lt(x, 0), pt.gt(x, n)),
        -pt.inf,
        pt.switch(pt.eq(x, 0), log_zero_prob, log_nonzero_prob),
    )


def cdf(x, psi, n, p):
    base_cdf = Binomial.cdf(x, n, p)
    zi_cdf = (1 - psi) + psi * base_cdf
    return cdf_bounds(zi_cdf, x, 0, n)


def ppf(q, psi, n, p):
    params = (psi, n, p)
    return find_ppf_discrete(q, mean(psi, n, p), 0, n, cdf, pdf, *params)


def sf(x, psi, n, p):
    return 1.0 - cdf(x, psi, n, p)


def isf(q, psi, n, p):
    return ppf(1.0 - q, psi, n, p)


def rvs(psi, n, p, size=None, random_state=None):
    next_rng, base_samples = pt.random.binomial(
        n, p, size=size, rng=random_state, return_next_rng=True
    )
    next_rng, mask = pt.random.bernoulli(psi, size=size, rng=next_rng, return_next_rng=True)
    return pt.cast(mask, "int64") * base_samples


def logcdf(x, psi, n, p):
    base_cdf = Binomial.cdf(x, n, p)
    result = pt.log1p(psi * (base_cdf - 1))
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.switch(pt.gt(x, n), 0.0, result),
    )


def logsf(x, psi, n, p):
    return pt.log1p(-cdf(x, psi, n, p))
