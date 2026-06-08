import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.math import gammaln


def mean(n, p):
    return n * p


def mode(n, p):
    return pt.floor((n + 1) * p)


def var(n, p):
    return n * p * (1 - p)


def std(n, p):
    return pt.sqrt(var(n, p))


def skewness(n, p):
    q = 1 - p
    return (q - p) / pt.sqrt(n * p * q)


def kurtosis(n, p):
    return (1 - 6 * p * (1 - p)) / (n * p * (1 - p))


def entropy(n, p):
    raise NotImplementedError("Entropy for Multinomial is not implemented yet.")


def pdf(x, n, p):
    return pt.exp(logpdf(x, n, p))


def logpdf(x, n, p):
    x = pt.as_tensor(x)
    p = pt.as_tensor(p)

    result = gammaln(n + 1) - pt.sum(gammaln(x + 1), axis=-1)
    result += pt.sum(x * pt.log(pt.maximum(p, 1e-10)), axis=-1)
    result = pt.switch(
        pt.or_(
            pt.any(pt.lt(x, 0), axis=-1),
            pt.neq(pt.sum(x, axis=-1), n),
        ),
        -np.inf,
        result,
    )

    return result


def rvs(n, p, size=None, random_state=None):
    return pt.random.multinomial(n, p, size=size, rng=random_state, return_next_rng=True)[1]
