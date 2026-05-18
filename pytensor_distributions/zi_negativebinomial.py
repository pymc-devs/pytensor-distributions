import pytensor.tensor as pt

from pytensor_distributions import negativebinomial as NegativeBinomial
from pytensor_distributions.helper import cdf_bounds, discrete_entropy, sf_bounds, zi_mode
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf_discrete


def mean(psi, n, p):
    return psi * n * (1 - p) / p


def mode(psi, n, p):
    base_mode = NegativeBinomial.mode(n, p)
    return zi_mode(base_mode, logpdf, psi, n, p)


def median(psi, n, p):
    return ppf(0.5, psi, n, p)


def var(psi, n, p):
    base_mean = n * (1 - p) / p
    base_var = n * (1 - p) / pt.power(p, 2)
    return psi * (base_var + (1 - psi) * pt.power(base_mean, 2))


def std(psi, n, p):
    return pt.sqrt(var(psi, n, p))


def skewness(psi, n, p):
    q = 1 - p
    ex1_fact = n * q / p
    ex2_fact = n * (n + 1) * pt.power(q, 2) / pt.power(p, 2)
    ex3_fact = n * (n + 1) * (n + 2) * pt.power(q, 3) / pt.power(p, 3)

    base_ex1 = ex1_fact
    base_ex2 = ex2_fact + ex1_fact
    base_ex3 = ex3_fact + 3 * ex2_fact + ex1_fact

    ex1 = psi * base_ex1
    ex2 = psi * base_ex2
    ex3 = psi * base_ex3

    mu_val = ex1
    mu2 = ex2 - pt.power(mu_val, 2)
    mu3 = ex3 - 3 * mu_val * ex2 + 2 * pt.power(mu_val, 3)

    return mu3 / pt.power(mu2, 1.5)


def kurtosis(psi, n, p):
    q = 1 - p
    ex1_fact = n * q / p
    ex2_fact = n * (n + 1) * pt.power(q, 2) / pt.power(p, 2)
    ex3_fact = n * (n + 1) * (n + 2) * pt.power(q, 3) / pt.power(p, 3)
    ex4_fact = n * (n + 1) * (n + 2) * (n + 3) * pt.power(q, 4) / pt.power(p, 4)

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
    lower = ppf(0.0001, psi, n, p)
    upper = ppf(0.9999, psi, n, p)
    return discrete_entropy(lower, upper, logpdf, psi, n, p)


def pdf(x, psi, n, p):
    result = pt.exp(logpdf(x, psi, n, p))
    return pt.switch(
        pt.isinf(x) & pt.gt(x, 0),
        pt.nan,
        result,
    )


def logpdf(x, psi, n, p):
    x = pt.as_tensor_variable(x)

    log_zero_prob = pt.log((1 - psi) + psi * pt.power(p, n))

    base_logpdf = NegativeBinomial.logpdf(x, n, p)
    log_nonzero_prob = pt.log(psi) + base_logpdf

    result = pt.switch(
        pt.or_(pt.lt(x, 0), pt.isinf(x)),
        -pt.inf,
        pt.switch(pt.eq(x, 0), log_zero_prob, log_nonzero_prob),
    )

    return pt.switch(
        pt.isinf(x) & pt.gt(x, 0),
        pt.nan,
        result,
    )


def cdf(x, psi, n, p):
    base_cdf = NegativeBinomial.cdf(x, n, p)
    zi_cdf = (1 - psi) + psi * base_cdf
    return cdf_bounds(zi_cdf, x, 0, pt.inf)


def ppf(q, psi, n, p):
    params = (psi, n, p)
    return find_ppf_discrete(q, mean(psi, n, p), 0, pt.inf, cdf, pdf, *params)


def sf(x, psi, n, p):
    base_sf = NegativeBinomial.sf(x, n, p)
    zi_sf = psi * base_sf
    return sf_bounds(zi_sf, x, 0, pt.inf)


def isf(q, psi, n, p):
    return ppf(1.0 - q, psi, n, p)


def rvs(psi, n, p, size=None, random_state=None):
    base_samples = pt.random.negative_binomial(n, p, size=size, rng=random_state)
    mask = pt.random.bernoulli(psi, size=size)
    return pt.cast(mask, "int64") * base_samples


def logcdf(x, psi, n, p):
    base_cdf = NegativeBinomial.cdf(x, n, p)
    result = pt.log1p(psi * (base_cdf - 1))
    return pt.switch(
        pt.or_(pt.lt(x, 0), pt.isinf(x)),
        pt.switch(pt.lt(x, 0), -pt.inf, 0.0),
        result,
    )


def logsf(x, psi, n, p):
    return pt.log(sf(x, psi, n, p))


def from_mu_alpha(psi, mu, alpha):
    n, p = NegativeBinomial.from_mu_alpha(mu, alpha)
    return psi, n, p


def to_mu_alpha(psi, n, p):
    mu, alpha = NegativeBinomial.to_mu_alpha(n, p)
    return psi, mu, alpha
