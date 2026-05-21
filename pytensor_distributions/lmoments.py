import numpy as np
import pytensor.tensor as pt


def _shifted_legendre(r: int, u):
    """
    Shifted Legendre polynomial P*_{r-1}(u) on [0, 1] for r = 1, 2, 3, 4.

    Parameters
    ----------
    r : int
        L-moment order (1-indexed).
    u : tensor
        Evaluation points in [0, 1].

    Notes
    -----
    .. math::
        P*_0(u) = 1
        P*_1(u) = 2u - 1
        P*_2(u) = 6u^2 - 6u + 1
        P*_3(u) = 20u^3 - 30u^2 + 12u - 1

    Returns
    -------
    tensor
    """
    if r == 1:
        return pt.ones_like(u)
    elif r == 2:
        return 2.0 * u - 1.0
    elif r == 3:
        return 6.0 * u**2 - 6.0 * u + 1.0
    elif r == 4:
        return 20.0 * u**3 - 30.0 * u**2 + 12.0 * u - 1.0
    else:
        raise NotImplementedError(f"Shifted Legendre polynomial not implemented for r={r}.")


# Pre-compute GL nodes/weights for common n_points values so repeated calls
# with the same n_points don't recompute them.
_GL_CACHE = {}


def _gl_nodes_weights(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights shifted to [0, 1]."""
    if n_points not in _GL_CACHE:
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        _GL_CACHE[n_points] = ((nodes + 1.0) / 2.0, weights / 2.0)
    return _GL_CACHE[n_points]


def _lmoment_from_ppf(ppf_func, params, r=None, n_points=50):
    """Compute the r-th L-moment by numerical integration of the PPF.

    Parameters
    ----------
    ppf_func : callable
        PPF (quantile function) with signature ppf(q, *params).
    params : sequence of tensor
        Distribution parameters.
    r : int
        L-moment order, 1 <= r <= 4.
    n_points : int
        Number of Gauss-Legendre quadrature nodes. Defaults to 50, which gives
        accuracy equivalent to the trapezoidal rule at ~5000+ points for smooth
        PPFs. Can typically be reduced to 20-30 without any loss in practice.

    Returns
    -------
    tensor
        The r-th L-moment.

    Notes
    -----
    L-moments are computed via integration of the quantile function (PPF) against
    shifted Legendre polynomials:

    .. math::
        lambda_r = int_0^1 Q(u) cdot P^*_{r-1}(u), du

    where :math:`P^*_{r-1}` are the shifted Legendre polynomials on [0, 1] and
    :math:`Q` = ppf.

    Gauss-Legendre quadrature is used instead of the trapezoidal rule because
    the integrand is smooth, allowing exponential convergence with far fewer
    PPF evaluations. Endpoint singularity padding (eps) is also unnecessary
    since GL nodes never fall on the boundary.

    References
    ----------
    .. [1] Hosking, J.R.M. *L-moments: analysis and estimation of distributions
        using linear combinations of order statistics*. 1990. Journal of the Royal
        Statistical Society, Series B. https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
    """
    np_nodes, np_weights = _gl_nodes_weights(n_points)
    nodes = pt.as_tensor_variable(np_nodes, dtype="float64")
    weights = pt.as_tensor_variable(np_weights, dtype="float64")

    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    # Reshape nodes/weights to broadcast over parameter dimensions:
    # (n_points, 1, ..., 1) vs params of shape (param_shape,)
    if broadcast_shape.ndim > 0:
        expand = (-1,) + (1,) * broadcast_shape.ndim
        nodes_bc = nodes.reshape(expand)
        weights_bc = weights.reshape(expand)
    else:
        nodes_bc = nodes
        weights_bc = weights

    quantiles = ppf_func(nodes_bc, *params)
    legendre_vals = _shifted_legendre(r, nodes_bc)

    result = pt.sum(weights_bc * quantiles * legendre_vals, axis=0)

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def _lmoments(ppf_func, *params, r=None, n_points=50):
    """Compute L-moments from 2 to 4 using the PPF and numerical integration.

    r1 is the mean, so we omit it.
    r2 is the second L-moment.
    For third and fourth L-moments, we use the ratios.
    tau3 = r3/r2 is the L-skewness
    tau4 = r4/r2 is the L-kurtosis
    """
    if r == 2:
        return _lmoment_from_ppf(ppf_func, params, r=2, n_points=n_points)
    if r == 3:
        l3 = _lmoment_from_ppf(ppf_func, params, r=3, n_points=n_points)
        l2 = _lmoment_from_ppf(ppf_func, params, r=2, n_points=n_points)
        return l3 / l2
    if r == 4:
        l4 = _lmoment_from_ppf(ppf_func, params, r=4, n_points=n_points)
        l2 = _lmoment_from_ppf(ppf_func, params, r=2, n_points=n_points)
        return l4 / l2
    else:
        raise NotImplementedError(f"L-moments not implemented for r={r}.")
