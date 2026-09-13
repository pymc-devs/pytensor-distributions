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


def mean(K, eta):
    return pt.eye(K)


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

    # Initialize the correlation matrix for K = 2
    beta_param = eta + (K - 2) / 2
    rng, u = pt.random.beta(
        beta_param, beta_param, size=batch_shape, rng=random_state, return_next_rng=True
    )
    r_val = 2.0 * u - 1.0
    ones = pt.ones(batch_shape)
    R = pt.stack([pt.stack([ones, r_val], axis=-1), pt.stack([r_val, ones], axis=-1)], axis=-2)

    for m in range(2, K):
        beta_param = beta_param - 0.5
        # Sample the beta radius
        rng, r_sq = pt.random.beta(
            m / 2, beta_param, size=batch_shape, rng=rng, return_next_rng=True
        )
        r = pt.sqrt(r_sq)

        # Sample a random point, uniformly distributed on the unit sphere
        rng, raw_normal = pt.random.normal(
            0, 1, size=batch_shape + (m,), rng=rng, return_next_rng=True
        )
        sphere_direction = raw_normal / pt.linalg.norm(raw_normal, ord=2, axis=-1, keepdims=True)
        # Create Target vector
        z = r[..., None] * sphere_direction

        # Transform z based on the cholesky factor
        L = pt.linalg.cholesky(R)
        y = pt.einsum("...ij,...j->...i", L, z)

        R = pt.concatenate(
            [
                pt.concatenate([R, y[..., None]], axis=-1),
                pt.concatenate([y[..., None, :], pt.ones(batch_shape + (1, 1))], axis=-1),
            ],
            axis=-2,
        )

    return R
