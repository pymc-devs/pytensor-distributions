import pytensor.tensor as pt


def logpdf(x, K, eta):
    k = pt.arange(1, K)
    beta_arg = eta + (K - k - 1) / 2
    log_norm_const = -(
        pt.sum((2 * (eta - 1) + K - k) * (K - k)) * pt.log(2)
        + pt.sum((K - k) * pt.special.betaln(beta_arg, beta_arg))
    )
    return log_norm_const + (eta - 1) * pt.linalg.slogdet(x)[1]


def pdf(C, K, eta):
    return pt.exp(logpdf(C, K, eta))


def entropy(K, eta):
    k = pt.arange(1, K)
    dk = K - k
    beta_k = eta + (dk - 1) / 2.0
    A = pt.sum((2.0 * (eta - 1.0) + dk) * dk) * pt.log(2.0) + pt.sum(
        dk * pt.special.betaln(beta_k, beta_k)
    )
    expected_logdet = pt.sum(dk * (pt.log(4.0) + 2.0 * (pt.psi(beta_k) - pt.psi(2.0 * beta_k))))
    return A - (eta - 1.0) * expected_logdet


def mean(K, eta):
    return pt.eye(K)


def median(K, eta):
    return pt.eye(K)


def mode(K, eta):
    alpha = eta - 1 + K / 2
    return pt.where(alpha > 1, pt.eye(K), pt.full((K, K), pt.nan))


def var(K, eta):
    return 1.0 / (2 * eta + K - 1)


def std(K, eta):
    return pt.sqrt(var(K, eta))


def rvs(K, eta, size=None, random_state=None):
    """Generate rvs using the onion method."""
    if size is None:
        batch_shape = ()
    elif isinstance(size, int):
        batch_shape = (size,)
    else:
        batch_shape = tuple(size)
    if K == 1:
        return pt.ones(batch_shape + (1, 1))

    # Initialize for K = 2
    beta = eta - 1.0 + K / 2.0
    rng, y = pt.random.beta(
        beta, beta, size=batch_shape, rng=random_state, return_next_rng=True
    )
    r = 2.0 * y - 1.0
    F = pt.full((*batch_shape, K, K), pt.eye(K))
    F = pt.set_subtensor(F[..., 0, 1], r)
    F = pt.set_subtensor(F[..., 1, 1], pt.sqrt(1.0 - r**2))
    for m in range(2, K):
        beta = beta - 0.5
        rng, y = pt.random.beta(
            m / 2.0, beta, size=batch_shape, rng=rng, return_next_rng=True
        )
        rng, z = pt.random.normal(
            0, 1, size=batch_shape + (m,), rng=rng, return_next_rng=True
        )
        z = z / pt.sqrt(pt.sum(z**2, axis=-1, keepdims=True))
        F = pt.set_subtensor(F[..., :m, m], pt.sqrt(y)[..., None] * z)
        F = pt.set_subtensor(F[..., m, m], pt.sqrt(1.0 - y))
    # Construct the correlation matrix from the factor
    return F.mT @ F
