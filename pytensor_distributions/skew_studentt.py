import pytensor.tensor as pt
from pytensor.tensor.math import betaincinv
from pytensor.tensor.special import betaln

from pytensor_distributions.helper import continuous_entropy, continuous_mode, ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def _raw_moment(n, a, b):
    """Compute the n-th raw moment of the standardized Jones-Faddy skew-t."""
    log_coeff = 0.5 * n * pt.log(a + b) - n * pt.log(2) - betaln(a, b)

    if n == 1:
        term0 = pt.exp(betaln(a + 0.5, b - 0.5))
        term1 = pt.exp(betaln(a - 0.5, b + 0.5))
        sum_terms = term0 - term1
    elif n == 2:
        term0 = pt.exp(betaln(a + 1, b - 1))
        term1 = 2 * pt.exp(betaln(a, b))
        term2 = pt.exp(betaln(a - 1, b + 1))
        sum_terms = term0 - term1 + term2
    elif n == 3:
        term0 = pt.exp(betaln(a + 1.5, b - 1.5))
        term1 = 3 * pt.exp(betaln(a + 0.5, b - 0.5))
        term2 = 3 * pt.exp(betaln(a - 0.5, b + 0.5))
        term3 = pt.exp(betaln(a - 1.5, b + 1.5))
        sum_terms = term0 - term1 + term2 - term3
    elif n == 4:
        term0 = pt.exp(betaln(a + 2, b - 2))
        term1 = 4 * pt.exp(betaln(a + 1, b - 1))
        term2 = 6 * pt.exp(betaln(a, b))
        term3 = 4 * pt.exp(betaln(a - 1, b + 1))
        term4 = pt.exp(betaln(a - 2, b + 2))
        sum_terms = term0 - term1 + term2 - term3 + term4
    else:
        raise ValueError(f"Moment order {n} not supported")

    return pt.exp(log_coeff) * sum_terms


def mean(a, b, mu, sigma):
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    mean_std = _raw_moment(1, a_b, b_b)
    result = mu_b + sigma_b * mean_std
    return pt.switch((a_b > 0.5) & (b_b > 0.5), result, pt.nan)


def mode(a, b, mu, sigma):
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    std_approx = sigma_b * pt.sqrt((a_b + b_b) / (2 * a_b * b_b))
    lower = mu_b - 5 * std_approx
    upper = mu_b + 5 * std_approx
    return continuous_mode(lower, upper, logpdf, a_b, b_b, mu_b, sigma_b)


def median(a, b, mu, sigma):
    return ppf(0.5, a, b, mu, sigma)


def var(a, b, mu, sigma):
    a_b, b_b, _, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    e_z = _raw_moment(1, a_b, b_b)
    e_z2 = _raw_moment(2, a_b, b_b)
    var_std = e_z2 - e_z**2
    result = sigma_b**2 * var_std
    return pt.switch((a_b > 1) & (b_b > 1), result, pt.nan)


def std(a, b, mu, sigma):
    variance = var(a, b, mu, sigma)
    return pt.switch(pt.isinf(variance) | pt.isnan(variance), variance, pt.sqrt(variance))


def skewness(a, b, mu, sigma):
    a_b, b_b, _, _ = pt.broadcast_arrays(a, b, mu, sigma)
    e_z = _raw_moment(1, a_b, b_b)
    e_z2 = _raw_moment(2, a_b, b_b)
    e_z3 = _raw_moment(3, a_b, b_b)
    var_z = e_z2 - e_z**2
    std_z = pt.sqrt(var_z)
    third_central = e_z3 - 3 * e_z * e_z2 + 2 * e_z**3
    result = third_central / std_z**3
    return pt.switch((a_b > 1.5) & (b_b > 1.5), result, pt.nan)


def kurtosis(a, b, mu, sigma):
    a_b, b_b, _, _ = pt.broadcast_arrays(a, b, mu, sigma)
    e_z = _raw_moment(1, a_b, b_b)
    e_z2 = _raw_moment(2, a_b, b_b)
    e_z3 = _raw_moment(3, a_b, b_b)
    e_z4 = _raw_moment(4, a_b, b_b)
    var_z = e_z2 - e_z**2
    fourth_central = e_z4 - 4 * e_z * e_z3 + 6 * e_z**2 * e_z2 - 3 * e_z**4
    result = fourth_central / var_z**2 - 3
    return pt.switch((a_b > 2) & (b_b > 2), result, pt.nan)


def lmoment1(a, b, mu, sigma):
    return mean(a, b, mu, sigma)


def lmoment2(a, b, mu, sigma):
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    return pt.switch((a_b > 0.5) & (b_b > 0.5), _lmoments(ppf, a, b, mu, sigma, r=2), pt.nan)


def lmoment3(a, b, mu, sigma):
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    return pt.switch((a_b > 0.5) & (b_b > 0.5), _lmoments(ppf, a, b, mu, sigma, r=3), pt.nan)


def lmoment4(a, b, mu, sigma):
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    return pt.switch((a_b > 0.5) & (b_b > 0.5), _lmoments(ppf, a, b, mu, sigma, r=4), pt.nan)


def entropy(a, b, mu, sigma):
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    lower = mu_b - 50 * sigma_b
    upper = mu_b + 50 * sigma_b
    return continuous_entropy(lower, upper, logpdf, a_b, b_b, mu_b, sigma_b)


def pdf(x, a, b, mu, sigma):
    return pt.exp(logpdf(x, a, b, mu, sigma))


def logpdf(x, a, b, mu, sigma):
    z = (x - mu) / sigma
    sqrt_term = pt.sqrt(a + b + z**2)
    a_term = (a + 0.5) * pt.log(1 + z / sqrt_term)
    b_term = (b + 0.5) * pt.log(1 - z / sqrt_term)
    norm_const = (a + b - 1) * pt.log(2) + betaln(a, b) + 0.5 * pt.log(a + b)
    result = a_term + b_term - norm_const - pt.log(sigma)
    return pt.switch(pt.isinf(x), pt.nan, result)


def cdf(x, a, b, mu, sigma):
    z = (x - mu) / sigma
    y = 0.5 * (1 + z / pt.sqrt(a + b + z**2))
    result = pt.betainc(a, b, y)
    result = pt.switch(pt.eq(x, -pt.inf), 0.0, result)
    result = pt.switch(pt.eq(x, pt.inf), 1.0, result)
    return result


def logcdf(x, a, b, mu, sigma):
    z = (x - mu) / sigma
    y = 0.5 * (1 + z / pt.sqrt(a + b + z**2))
    result = pt.log(pt.betainc(a, b, y))
    result = pt.switch(pt.eq(x, -pt.inf), -pt.inf, result)
    result = pt.switch(pt.eq(x, pt.inf), 0.0, result)
    return result


def sf(x, a, b, mu, sigma):
    # uses symmetry: sf(x; a, b) = cdf(-x; b, a)
    return cdf(-x, b, a, -mu, sigma)


def logsf(x, a, b, mu, sigma):
    return logcdf(-x, b, a, -mu, sigma)


def ppf(q, a, b, mu, sigma):
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    bval = betaincinv(a_b, b_b, q)
    num = (2 * bval - 1) * pt.sqrt(a_b + b_b)
    denom = 2 * pt.sqrt(bval * (1 - bval))
    z = num / denom
    result = mu_b + sigma_b * z
    return ppf_bounds_cont(result, q, -pt.inf, pt.inf)


def isf(q, a, b, mu, sigma):
    return ppf(1 - q, a, b, mu, sigma)


def rvs(a, b, mu, sigma, size=None, random_state=None):
    u = pt.random.uniform(0, 1, size=size, rng=random_state)
    return ppf(u, a, b, mu, sigma)
