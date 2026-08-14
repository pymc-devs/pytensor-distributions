import pytensor.tensor as pt
from pytensor.tensor.math import gammaln


def mean(nu, Psi):
    Psi = pt.as_tensor(Psi)
    p = int(Psi.type.shape[-1])
    return pt.switch(nu > p + 1, Psi / (nu - p - 1), pt.nan)


def mode(nu, Psi):
    Psi = pt.as_tensor(Psi)
    p = int(Psi.type.shape[-1])
    return Psi / (nu + p + 1)


def var(nu, Psi):
    Psi = pt.as_tensor(Psi)
    p = int(Psi.type.shape[-1])
    diag_Psi = pt.diagonal(Psi, axis1=-2, axis2=-1)
    return pt.switch(
        nu > p + 3,
        2 * diag_Psi ** 2 / ((nu - p - 1) ** 2 * (nu - p - 3)),
        pt.nan,
    )


def std(nu, Psi):
    return pt.sqrt(var(nu, Psi))


def entropy(nu, Psi):
    Psi = pt.as_tensor(Psi)
    p = int(Psi.type.shape[-1])
    _, logdet_Psi = pt.linalg.slogdet(Psi)

    psi_eval_points = pt.stack(
        [(nu - p + i) / 2 for i in range(1, p + 1)]
    )

    mvgammaln = (
        0.25 * p * (p - 1) * pt.log(pt.pi)
        + pt.sum(
            pt.stack(
                [gammaln((nu + 1 - i) / 2) for i in range(1, p + 1)]
            )
        )
    )

    return (
        mvgammaln
        + 0.5 * p * nu
        + 0.5 * (p + 1) * (logdet_Psi - pt.log(2))
        - 0.5 * (nu + p + 1) * pt.sum(pt.digamma(psi_eval_points))
    )


def pdf(X, nu, Psi):
    return pt.exp(logpdf(X, nu, Psi))


def logpdf(X, nu, Psi):
    X = pt.as_tensor(X)
    Psi = pt.as_tensor(Psi)

    p = int(Psi.type.shape[-1])

    _, logdet_X = pt.linalg.slogdet(X)
    _, logdet_Psi = pt.linalg.slogdet(Psi)

    X_inv = pt.linalg.inv(X)
    trace_term = pt.sum(Psi * X_inv, axis=(-2, -1))

    log_gamma_p = (
        0.25 * p * (p - 1) * pt.log(pt.pi)
        + pt.sum(
            pt.stack(
                [
                    gammaln((nu + 1 - i) / 2)
                    for i in range(1, p + 1)
                ]
            ),
            axis=0,
        )
    )

    result = (
        0.5 * nu * logdet_Psi
        - 0.5 * nu * p * pt.log(2)
        - log_gamma_p
        - 0.5 * (nu + p + 1) * logdet_X
        - 0.5 * trace_term
    )

    result = pt.switch(nu <= p - 1, -pt.inf, result)

    return result


def rvs(nu, Psi, size=None, random_state=None):
    Psi = pt.as_tensor(Psi)
    p = int(Psi.type.shape[-1])
    Psi_inv = pt.linalg.inv(Psi)
    L = pt.linalg.cholesky(Psi_inv, lower=True)

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
    X = pt.linalg.inv(W)
    if squeeze:
        return X[0]
    return X
