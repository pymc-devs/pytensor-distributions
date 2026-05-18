import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, discrete_entropy
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf_discrete


def mean(n, alpha, beta):
    return n * alpha / (alpha + beta)


def mode(n, alpha, beta):
    # The mode depends on the shape of the distribution:
    # - alpha > 1, beta > 1: unimodal, standard formula applies
    # - alpha = 1, beta > 1: monotonically decreasing, mode is 0
    # - alpha > 1, beta = 1: monotonically increasing, mode is n
    # - alpha = 1, beta = 1: uniform, no unique mode (return NaN)
    # - alpha < 1 or beta < 1 (other cases): U-shaped or J-shaped, no unique mode (return NaN)
    # This follows the same convention as distributions/beta.py
    n_b, alpha_b, beta_b = pt.broadcast_arrays(n, alpha, beta)
    result = pt.full_like(alpha_b, pt.nan, dtype="float64")

    # Monotonically decreasing: alpha = 1 and beta > 1 -> mode is 0
    result = pt.where(pt.eq(alpha_b, 1) & pt.gt(beta_b, 1), 0.0, result)
    # Monotonically increasing: alpha > 1 and beta = 1 -> mode is n
    result = pt.where(pt.gt(alpha_b, 1) & pt.eq(beta_b, 1), n_b, result)
    # Standard unimodal case: alpha > 1 and beta > 1
    standard_mode = pt.floor((n_b + 1) * ((alpha_b - 1) / (alpha_b + beta_b - 2)))
    result = pt.where(
        pt.gt(alpha_b, 1) & pt.gt(beta_b, 1),
        pt.clip(standard_mode, 0, n_b),
        result,
    )

    return result


def median(n, alpha, beta):
    return ppf(0.5, n, alpha, beta)


def var(n, alpha, beta):
    return n * alpha * beta * (alpha + beta + n) / ((alpha + beta) ** 2 * (alpha + beta + 1))


def std(n, alpha, beta):
    return pt.sqrt(var(n, alpha, beta))


def skewness(n, alpha, beta):
    return (
        (alpha + beta + 2 * n)
        * (beta - alpha)
        / (alpha + beta + 2)
        * pt.sqrt((1 + alpha + beta) / (n * alpha * beta * (alpha + beta + n)))
    )


def kurtosis(n, alpha, beta):
    alpha_beta_sum = alpha + beta
    alpha_beta_product = alpha * beta
    numerator = ((alpha_beta_sum) ** 2) * (1 + alpha_beta_sum)
    denominator = (
        (n * alpha_beta_product)
        * (alpha_beta_sum + 2)
        * (alpha_beta_sum + 3)
        * (alpha_beta_sum + n)
    )
    left = numerator / denominator
    right = (
        (alpha_beta_sum) * (alpha_beta_sum - 1 + 6 * n)
        + 3 * alpha_beta_product * (n - 2)
        + 6 * n**2
    )
    right -= (3 * alpha_beta_product * n * (6 - n)) / alpha_beta_sum
    right -= (18 * alpha_beta_product * n**2) / (alpha_beta_sum) ** 2
    return (left * right) - 3


def lmoment1(n, alpha, beta):
    return mean(n, alpha, beta)


def lmoment2(n, alpha, beta):
    return _lmoments(ppf, n, alpha, beta, r=2)


def lmoment3(n, alpha, beta):
    return _lmoments(ppf, n, alpha, beta, r=3)


def lmoment4(n, alpha, beta):
    return _lmoments(ppf, n, alpha, beta, r=4)


def entropy(n, alpha, beta):
    return discrete_entropy(0, n + 1, logpdf, n, alpha, beta)


def pdf(x, n, alpha, beta):
    return pt.exp(logpdf(x, n, alpha, beta))


def cdf(x, n, alpha, beta):
    broadcast_shape = pt.broadcast_arrays(x, n, alpha, beta)[0]
    k_vals = pt.arange(0, pt.max(x) + 1)
    k_broadcast = k_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)

    prob = pt.sum(
        pt.where(pt.le(k_broadcast, pt.floor(x)), pdf(k_broadcast, n, alpha, beta), 0.0), axis=0
    )
    return cdf_bounds(prob, x, 0, n)


def ppf(q, n, alpha, beta):
    params = (n, alpha, beta)
    return find_ppf_discrete(q, mean(n, alpha, beta), 0, n, cdf, pdf, *params)


def sf(x, n, alpha, beta):
    return 1 - cdf(x, n, alpha, beta)


def isf(x, n, alpha, beta):
    return ppf(1 - x, n, alpha, beta)


def rvs(n, alpha, beta, size=None, random_state=None):
    return pt.random.betabinom(n, alpha, beta, size=size, rng=random_state)


def logcdf(x, n, alpha, beta):
    return pt.log(cdf(x, n, alpha, beta))


def logsf(x, n, alpha, beta):
    return pt.log(sf(x, n, alpha, beta))


def logpdf(x, n, alpha, beta):
    return pt.switch(
        pt.or_(pt.lt(x, 0), pt.gt(x, n)),
        -pt.inf,
        (
            pt.gammaln(n + 1)
            - pt.gammaln(x + 1)
            - pt.gammaln(n - x + 1)
            + pt.gammaln(x + alpha)
            + pt.gammaln(n - x + beta)
            - pt.gammaln(n + alpha + beta)
            + pt.gammaln(alpha + beta)
            - pt.gammaln(alpha)
            - pt.gammaln(beta)
        ),
    )
