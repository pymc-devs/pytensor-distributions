import pytensor.tensor as pt

from pytensor_distributions.helper import cdf_bounds, discrete_entropy
from pytensor_distributions.lmoments import _lmoments
from pytensor_distributions.optimization import find_ppf_discrete


def _support_lower(N, k, n):
    return pt.maximum(0, n + k - N)


def _support_upper(N, k, n):
    return pt.minimum(k, n)


def _log_binomial(n, k):
    return pt.gammaln(n + 1) - pt.gammaln(k + 1) - pt.gammaln(n - k + 1)


def mean(N, k, n):
    return n * k / N


def mode(N, k, n):
    return pt.floor((n + 1) * (k + 1) / (N + 2))


def median(N, k, n):
    return ppf(0.5, N, k, n)


def var(N, k, n):
    return n * k * (N - k) * (N - n) / (N * N * (N - 1))


def std(N, k, n):
    return pt.sqrt(var(N, k, n))


def skewness(N, k, n):
    numerator = (N - 2 * k) * pt.sqrt(N - 1) * (N - 2 * n)
    denominator = pt.sqrt(n * k * (N - k) * (N - n)) * (N - 2)
    return numerator / denominator


def kurtosis(N, k, n):
    N = pt.cast(N, "float64")
    k = pt.cast(k, "float64")
    n = pt.cast(n, "float64")
    m = N - k
    num = N * N * (N - 1) * (N * (N + 1) - 6 * n * (N - n) - 6 * k * m) + 6 * k * n * (
        N - n
    ) * m * (5 * N - 6)
    den = k * n * (N - n) * m * (N - 2) * (N - 3)
    return num / den


def lmoment1(N, k, n):
    return mean(N, k, n)


def lmoment2(N, k, n):
    return _lmoments(ppf, N, k, n, r=2)


def lmoment3(N, k, n):
    return _lmoments(ppf, N, k, n, r=3)


def lmoment4(N, k, n):
    return _lmoments(ppf, N, k, n, r=4)


def entropy(N, k, n):
    lower = _support_lower(N, k, n)
    upper = _support_upper(N, k, n) + 1
    return discrete_entropy(lower, upper, logpdf, N, k, n)


def pdf(x, N, k, n):
    return pt.exp(logpdf(x, N, k, n))


def logpdf(x, N, k, n):
    x = pt.floor(x)
    lower = _support_lower(N, k, n)
    upper = _support_upper(N, k, n)
    in_support = pt.and_(pt.ge(x, lower), pt.le(x, upper))
    result = _log_binomial(k, x) + _log_binomial(N - k, n - x) - _log_binomial(N, n)
    return pt.switch(in_support, result, -pt.inf)


def cdf(x, N, k, n):
    lower = _support_lower(N, k, n)
    upper = _support_upper(N, k, n)
    x = pt.as_tensor_variable(x)
    x_floor = pt.floor(x)
    safe_x = pt.clip(x_floor, 0, upper)
    safe_x = pt.switch(pt.isnan(x), 0, safe_x)
    x_vals = pt.arange(0, pt.cast(upper + 1, "int64"))
    log_pmf_vals = logpdf(x_vals, N, k, n)
    pmf_vals = pt.exp(log_pmf_vals)
    cumsum = pt.cumsum(pmf_vals)
    x_idx = pt.cast(safe_x, "int64")
    raw_cdf = cumsum[x_idx]
    raw_cdf = pt.switch(pt.isnan(x), pt.nan, raw_cdf)
    return cdf_bounds(raw_cdf, x, lower, upper)


def logcdf(x, N, k, n):
    return pt.log(cdf(x, N, k, n))


def sf(x, N, k, n):
    return 1.0 - cdf(x, N, k, n)


def logsf(x, N, k, n):
    return pt.log1p(-cdf(x, N, k, n))


def ppf(q, N, k, n):
    lower = _support_lower(N, k, n)
    upper = _support_upper(N, k, n)
    return find_ppf_discrete(q, mean(N, k, n), lower, upper, cdf, pdf, N, k, n)


def isf(q, N, k, n):
    return ppf(1.0 - q, N, k, n)


def rvs(N, k, n, size=None, random_state=None):
    return pt.random.hypergeometric(k, N - k, n, size=size, rng=random_state)
