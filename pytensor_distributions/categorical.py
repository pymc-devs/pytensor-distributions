import pytensor.tensor as pt
from pytensor.tensor.xlogx import xlogx

from pytensor_distributions.helper import cdf_bounds, ppf_bounds_disc


def _normalize_p(p):
    p = pt.as_tensor_variable(p)
    return p / pt.sum(p, axis=-1, keepdims=True)


def _k(p):
    return pt.shape(p)[-1]


def _central_moment(p, order):
    p = _normalize_p(p)
    k = _k(p)
    indices = pt.arange(k)
    mu = pt.sum(indices * p, axis=-1, keepdims=True)
    return pt.sum((indices - mu) ** order * p, axis=-1)


def mean(p):
    p = _normalize_p(p)
    indices = pt.arange(_k(p))
    return pt.sum(indices * p, axis=-1)


def mode(p):
    p = _normalize_p(p)
    return pt.argmax(p, axis=-1)


def median(p):
    return ppf(0.5, p)


def var(p):
    return _central_moment(p, 2)


def std(p):
    return pt.sqrt(var(p))


def skewness(p):
    return _central_moment(p, 3) / var(p) ** 1.5


def kurtosis(p):
    return _central_moment(p, 4) / var(p) ** 2 - 3


def entropy(p):
    p = _normalize_p(p)
    return -pt.sum(xlogx(p), axis=-1)


def pdf(x, p):
    return pt.exp(logpdf(x, p))


def logpdf(x, p):
    p = _normalize_p(p)
    k = _k(p)
    x = pt.as_tensor_variable(x)
    x_int = pt.cast(x, "int64")
    in_support = pt.and_(pt.ge(x, 0), pt.lt(x, k))
    is_integer = pt.eq(x, pt.floor(x))
    valid = pt.and_(in_support, is_integer)
    safe_x = pt.clip(x_int, 0, k - 1)
    log_p = pt.log(p)[safe_x]
    return pt.switch(valid, log_p, -pt.inf)


def cdf(x, p):
    p = _normalize_p(p)
    k = _k(p)
    x = pt.as_tensor_variable(x)
    x_floor = pt.floor(x)
    cumsum_p = pt.cumsum(p, axis=-1)
    safe_x = pt.cast(pt.clip(x_floor, 0, k - 1), "int64")
    cdf_val = cumsum_p[safe_x]
    return cdf_bounds(cdf_val, x, 0, k - 1)


def logcdf(x, p):
    return pt.log(cdf(x, p))


def sf(x, p):
    return 1.0 - cdf(x, p)


def logsf(x, p):
    return pt.log1p(-cdf(x, p))


def ppf(q, p):
    p = _normalize_p(p)
    k = _k(p)
    q = pt.as_tensor_variable(q)
    cumsum_p = pt.cumsum(p, axis=-1)
    ge_mask = pt.ge(cumsum_p, pt.shape_padright(q))
    masked_indices = pt.switch(ge_mask, pt.arange(k), k)
    result = pt.min(masked_indices, axis=-1)
    return ppf_bounds_disc(result, q, 0, k - 1)


def isf(q, p):
    return ppf(1.0 - q, p)


def rvs(p, size=None, random_state=None):
    p = _normalize_p(p)
    return pt.random.categorical(p, size=size, rng=random_state, return_next_rng=True)[1]
