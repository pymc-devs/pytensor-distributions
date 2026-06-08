import pytensor.tensor as pt
from pytensor.tensor.math import gammaln


def mean(nu, V):
    return nu * V


def mode(nu, V):
    p = V.shape[-1]
    return pt.switch(nu >= p + 1, (nu - p - 1) * V, pt.nan)


def var(nu, V):
    V = pt.as_tensor(V)
    diag_V = pt.diagonal(V, axis1=-2, axis2=-1)
    return 2 * nu * diag_V**2


def std(nu, V):
    return pt.sqrt(var(nu, V))


def entropy(nu, V):
    V = pt.as_tensor(V)
    p = int(V.type.shape[-1])
    _, logdet_V = pt.linalg.slogdet(V)

    mvdigamma = pt.sum(pt.stack([pt.digamma((nu + 1 - i) / 2) for i in range(1, p + 1)]), axis=0)
    mvgammaln = 0.25 * p * (p - 1) * pt.log(pt.pi) + pt.sum(
        pt.stack([gammaln((nu + 1 - i) / 2) for i in range(1, p + 1)]), axis=0
    )

    exp_logdet_X = mvdigamma + p * pt.log(2) + logdet_V

    return (
        0.5 * nu * logdet_V
        + 0.5 * nu * p * pt.log(2)
        + mvgammaln
        - 0.5 * (nu - p - 1) * exp_logdet_X
        + 0.5 * nu * p
    )


def pdf(X, nu, V):
    return pt.exp(logpdf(X, nu, V))


def logpdf(X, nu, V):
    X = pt.as_tensor(X)
    V = pt.as_tensor(V)
    p = int(V.type.shape[-1])
    _, logdet_X = pt.linalg.slogdet(X)
    _, logdet_V = pt.linalg.slogdet(V)
    V_inv = pt.linalg.inv(V)
    trace_term = pt.sum(V_inv * X, axis=(-2, -1))
    log_gamma_p = 0.25 * p * (p - 1) * pt.log(pt.pi) + pt.sum(
        pt.stack([gammaln((nu + 1 - i) / 2) for i in range(1, p + 1)]), axis=0
    )
    result = (
        -log_gamma_p
        - 0.5 * nu * logdet_V
        - 0.5 * nu * p * pt.log(2)
        + 0.5 * (nu - p - 1) * logdet_X
        - 0.5 * trace_term
    )
    result = pt.switch(nu <= p - 1, -pt.inf, result)

    return result


def rvs(nu, V, size=None, random_state=None):
    V = pt.as_tensor(V)
    p = int(V.type.shape[-1])
    L = pt.linalg.cholesky(V, lower=True)

    if size is None:
        batch_size = 1
        squeeze = True
    else:
        batch_size = size if isinstance(size, int) else size
        squeeze = False

    chi_samples = pt.stack(
        [
            pt.sqrt(
                pt.random.chisquare(
                    nu - i, size=batch_size, rng=random_state, return_next_rng=True
                )[1]
            )
            for i in range(p)
        ],
        axis=1,
    )

    n_tril = p * (p - 1) // 2
    norm_samples = pt.random.normal(
        0, 1, size=(batch_size, n_tril), rng=random_state, return_next_rng=True
    )[1]
    A = pt.zeros((batch_size, p, p))

    diag_idx = pt.arange(p)
    A = pt.set_subtensor(A[:, diag_idx, diag_idx], chi_samples)

    tril_indices = pt.tril_indices(p, k=-1)
    A = pt.set_subtensor(A[:, tril_indices[0], tril_indices[1]], norm_samples)

    LA = L @ A
    W = LA @ pt.swapaxes(LA, -2, -1)

    if squeeze:
        return W[0]
    return W
