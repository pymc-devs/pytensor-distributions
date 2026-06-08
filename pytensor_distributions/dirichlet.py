import pytensor.tensor as pt
from pytensor.tensor.math import gammaln

from pytensor_distributions import beta


def mean(alpha):
    alpha_sum = pt.sum(alpha, axis=-1, keepdims=True)
    beta_param = alpha_sum - alpha
    return beta.mean(alpha, beta_param)


def mode(alpha):
    alpha = pt.as_tensor(alpha)
    k = alpha.shape[-1]
    alpha_sum = pt.sum(alpha, axis=-1, keepdims=True)
    mode_val = (alpha - 1) / (alpha_sum - k)
    return pt.switch(pt.all(alpha > 1, axis=-1, keepdims=True), mode_val, pt.nan)


def var(alpha):
    alpha_sum = pt.sum(alpha, axis=-1, keepdims=True)
    beta_param = alpha_sum - alpha
    return beta.var(alpha, beta_param)


def std(alpha):
    return pt.sqrt(var(alpha))


def skewness(alpha):
    alpha_sum = pt.sum(alpha, axis=-1, keepdims=True)
    beta_param = alpha_sum - alpha
    return beta.skewness(alpha, beta_param)


def kurtosis(alpha):
    alpha_sum = pt.sum(alpha, axis=-1, keepdims=True)
    beta_param = alpha_sum - alpha
    return beta.kurtosis(alpha, beta_param)


def entropy(alpha):
    alpha_sum = pt.sum(alpha, axis=-1)
    k = alpha.shape[-1]
    log_B = pt.sum(gammaln(alpha), axis=-1) - gammaln(alpha_sum)
    return (
        log_B
        + (alpha_sum - k) * pt.digamma(alpha_sum)
        - pt.sum((alpha - 1) * pt.digamma(alpha), axis=-1)
    )


def pdf(x, alpha):
    return pt.exp(logpdf(x, alpha))


def logpdf(x, alpha):
    x = pt.as_tensor(x)
    alpha = pt.as_tensor(alpha)
    res = pt.sum((alpha - 1) * pt.log(x) - gammaln(alpha), axis=-1) + gammaln(
        pt.sum(alpha, axis=-1)
    )
    res = pt.switch(
        pt.or_(
            pt.any(pt.lt(x, 0), axis=-1),
            pt.any(pt.gt(x, 1), axis=-1),
        ),
        -pt.inf,
        res,
    )
    return res


def rvs(alpha, size=None, random_state=None):
    return pt.random.dirichlet(alpha, size=size, rng=random_state, return_next_rng=True)[1]
