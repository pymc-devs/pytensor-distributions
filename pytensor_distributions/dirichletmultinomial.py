import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.math import gammaln

from pytensor_distributions import betabinomial


def mean(n, a):
    a = pt.as_tensor(a)
    a_sum = pt.sum(a, axis=-1, keepdims=True)
    beta = a_sum - a
    return betabinomial.mean(n, a, beta)


def mode(n, a):
    k = a.shape[-1]
    a_sum = pt.sum(a, axis=-1, keepdims=True)
    mode_val = pt.floor((n - k + 1) * a / (a_sum - k))
    valid = pt.and_(pt.all(a > 1, axis=-1, keepdims=True), n > k - 1)
    return pt.switch(valid, mode_val, pt.nan)


def var(n, a):
    a = pt.as_tensor(a)
    a_sum = pt.sum(a, axis=-1, keepdims=True)
    beta = a_sum - a
    return betabinomial.var(n, a, beta)


def std(n, a):
    return pt.sqrt(var(n, a))


def skewness(n, a):
    a = pt.as_tensor(a)
    a_sum = pt.sum(a, axis=-1, keepdims=True)
    beta = a_sum - a
    return betabinomial.skewness(n, a, beta)


def kurtosis(n, a):
    a = pt.as_tensor(a)
    a_sum = pt.sum(a, axis=-1, keepdims=True)
    beta = a_sum - a
    return betabinomial.kurtosis(n, a, beta)


def entropy(n, a):
    raise NotImplementedError("Entropy for Dirichlet-Multinomial is not implemented yet.")


def pdf(x, n, a):
    return pt.exp(logpdf(x, n, a))


def logpdf(x, n, a):
    x = pt.as_tensor(x)
    a = pt.as_tensor(a)
    a_sum = pt.sum(a, axis=-1)

    const = (gammaln(n + 1) + gammaln(a_sum)) - gammaln(n + a_sum)
    series = gammaln(x + a) - (gammaln(x + 1) + gammaln(a))
    res = const + pt.sum(series, axis=-1)

    res = pt.switch(
        pt.or_(
            pt.any(pt.lt(x, 0), axis=-1),
            pt.neq(pt.sum(x, axis=-1), n),
        ),
        -np.inf,
        res,
    )

    return res


def rvs(n, a, size=None, random_state=None):
    next_rng, p = pt.random.dirichlet(a, size=size, rng=random_state, return_next_rng=True)
    return pt.random.multinomial(n, p, rng=next_rng, return_next_rng=True)[1]
