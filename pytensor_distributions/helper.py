import numpy as np
import pytensor.tensor as pt

LOG2 = np.log(2.0)
SQRT2 = 2.0**0.5


def cdf_bounds(prob, x, lower, upper):
    """
    Apply bounds checking for cumulative distribution function (CDF).

    Parameters
    ----------
    prob : tensor
        The computed CDF probability value
    x : tensor
        Input value to evaluate CDF at
    lower : float
        Lower bound of the distribution support
    upper : float
        Upper bound of the distribution support

    Returns
    -------
    tensor
        CDF value with proper bounds: 0.0 for x < lower,
        1.0 for x >= upper, otherwise the computed prob value
    """
    x = pt.as_tensor_variable(x)
    return pt.switch(pt.lt(x, lower), 0.0, pt.switch(pt.ge(x, upper), 1.0, prob))


def ppf_bounds_cont(x_val, q, lower, upper):
    """
    Apply bounds checking for the inverse CDF of continuous distributions.

    Parameters
    ----------
    x_val : tensor
        The computed PPF value
    q : tensor
        Probability value (quantile) between 0 and 1
    lower : float
        Lower bound of the distribution support
    upper : float
        Upper bound of the distribution support

    Returns
    -------
    tensor
        PPF value with proper bounds: NaN for q outside [0,1],
        lower bound for q=0, upper bound for q=1, otherwise x_val
    """
    return pt.switch(
        pt.or_(pt.lt(q, 0), pt.gt(q, 1)),
        pt.nan,
        pt.switch(pt.eq(q, 0), lower, pt.switch(pt.eq(q, 1), upper, x_val)),
    )


def isf_bounds_cont(x_val, q, lower, upper):
    return ppf_bounds_cont(x_val, q, upper, lower)


def ppf_bounds_disc(x_val, q, lower, upper):
    """
    Apply bounds checking for the inverse CDF of discrete distributions.

    Parameters
    ----------
    x_val : tensor
        The computed PPF value
    q : tensor
        Probability value (quantile) between 0 and 1
    lower : int
        Lower bound of the distribution support
    upper : int
        Upper bound of the distribution support

    Returns
    -------
    tensor
        PPF value with proper bounds: NaN for q outside [0,1],
        lower-1 for q=0 (since discrete PPF at 0 is below support),
        upper for q=1, otherwise x_val
    """
    return pt.switch(
        pt.lt(q, 0),
        pt.nan,
        pt.switch(
            pt.gt(q, 1),
            pt.nan,
            pt.switch(pt.eq(q, 0), lower - 1, pt.switch(pt.eq(q, 1), upper, x_val)),
        ),
    )


def sf_bounds(x_val, q, lower, upper):
    return pt.switch(pt.lt(q, lower), 1.0, pt.switch(pt.gt(q, upper), 0.0, x_val))


def logdiffexp(a, b):
    """Return log(exp(a) - exp(b))."""
    return a + pt.log1mexp(b - a)


def continuous_entropy(min_x, max_x, logpdf_func, *params):
    """
    Compute entropy for continuous distributions using numerical integration.

    Parameters
    ----------
    min_x : float or tensor
        Minimum value for integration (can be batched)
    max_x : float or tensor
        Maximum value for integration (can be batched)
    logpdf_func : function
        Log probability density function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to logpdf_func

    Returns
    -------
    entropy : tensor
    """
    n_points = 1000

    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    x_values = pt.linspace(min_x, max_x, n_points)

    # When bounds are batched, linspace returns (n_points, batch_dims...) which
    # already broadcasts correctly with params. Only reshape for scalar bounds.
    if x_values.ndim == 1 and broadcast_shape.ndim > 0:
        x_broadcast = x_values.reshape((-1,) + (1,) * broadcast_shape.ndim)
    else:
        x_broadcast = x_values

    logpdf_vals = logpdf_func(x_broadcast, *params)
    pdf_vals = pt.exp(logpdf_vals)

    integrand = -pdf_vals * logpdf_vals

    dx = (max_x - min_x) / (n_points - 1)
    result = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1], axis=0) + 0.5 * integrand[-1])

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def discrete_entropy(min_x, max_x, logpdf, *params):
    """
    Compute entropy for discrete distributions by explicit summation.

    Parameters
    ----------
    min_x : int or float
        Minimum value to sum over
    max_x : int or float
        Maximum value to sum over
    logpdf : function
        Log probability mass function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to logpdf

    Returns
    -------
    entropy : tensor
    """
    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    k_vals = pt.arange(pt.min(min_x), pt.max(max_x))
    k_broadcast = k_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)

    log_probs = logpdf(k_broadcast, *params)
    # Zero out k values outside a column's support (log_probs == -inf) to avoid 0 * -inf = nan
    result = pt.sum(-pt.switch(pt.isinf(log_probs), 0.0, pt.exp(log_probs) * log_probs), axis=0)

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def discrete_moment(ppf, pdf, *params, order=1):
    """
    Compute central moments for discrete distributions by explicit summation.

    Parameters
    ----------
    ppf : function
        Percent-point function that takes (p, *params) as arguments
    pdf : function
        Probability mass function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to pdf
    order : int
        Order of the moment to compute

    Returns
    -------
    moment : tensor
    """
    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    min_x = ppf(0.0001, *params).min()
    max_x = ppf(0.9999, *params).max() + 1

    k_vals = pt.arange(min_x, max_x)
    k_broadcast = k_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)
    pdf_vals = pdf(k_broadcast, *params)

    result = pt.sum(k_broadcast * pdf_vals, axis=0)
    if order > 1:
        result = pt.sum((k_broadcast - result) ** order * pdf_vals, axis=0)

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def discrete_mean(ppf, pdf, *params):
    """Compute mean for discrete distributions."""
    return discrete_moment(ppf, pdf, *params, order=1)


def discrete_variance(ppf, pdf, *params):
    """Compute variance for discrete distributions."""
    return discrete_moment(ppf, pdf, *params, order=2)


def discrete_skewness(ppf, pdf, *params):
    """Compute skewness for discrete distributions."""
    variance = discrete_moment(ppf, pdf, *params, order=2)
    third_moment = discrete_moment(ppf, pdf, *params, order=3)
    return third_moment / (variance**1.5)


def discrete_kurtosis(ppf, pdf, *params):
    """Compute kurtosis for discrete distributions."""
    variance = discrete_moment(ppf, pdf, *params, order=2)
    fourth_moment = discrete_moment(ppf, pdf, *params, order=4)
    result = fourth_moment / (variance**2) - 3
    return result


def continuous_moment(lower, upper, logpdf, *params, order=1, mean_val=None, n_points=1000):
    """
    Compute raw or central moments for continuous distributions using numerical integration.

    Uses the trapezoidal rule for numerical integration.

    Parameters
    ----------
    lower : float or tensor
        Lower bound for integration (can be batched)
    upper : float or tensor
        Upper bound for integration (can be batched)
    logpdf : function
        Log probability density function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to logpdf
    order : int
        Order of the moment to compute
    mean_val : tensor, optional
        If provided, computes central moment around this mean.
        If None, computes raw moment.
    n_points : int
        Number of integration points

    Returns
    -------
    moment : tensor
    """
    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    x_vals = pt.linspace(lower, upper, n_points)

    # When bounds are batched, linspace returns (n_points, batch_dims...) which
    # already broadcasts correctly with params. Only reshape for scalar bounds.
    if x_vals.ndim == 1 and broadcast_shape.ndim > 0:
        x_broadcast = x_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)
    else:
        x_broadcast = x_vals
    pdf_vals = pt.exp(logpdf(x_broadcast, *params))

    if mean_val is not None:
        # Central moment
        integrand = (x_broadcast - mean_val) ** order * pdf_vals
    else:
        # Raw moment
        integrand = x_broadcast**order * pdf_vals

    dx = (upper - lower) / (n_points - 1)
    result = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1], axis=0) + 0.5 * integrand[-1])

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def continuous_mean(lower, upper, logpdf, *params):
    """Compute mean for continuous distributions."""
    return continuous_moment(lower, upper, logpdf, *params, order=1)


def continuous_variance(lower, upper, logpdf, *params):
    """Compute variance for continuous distributions."""
    mean_val = continuous_moment(lower, upper, logpdf, *params, order=1)
    return continuous_moment(lower, upper, logpdf, *params, order=2, mean_val=mean_val)


def continuous_skewness(lower, upper, logpdf, *params):
    """Compute skewness for continuous distributions."""
    mean_val = continuous_moment(lower, upper, logpdf, *params, order=1)
    variance = continuous_moment(lower, upper, logpdf, *params, order=2, mean_val=mean_val)
    third_central = continuous_moment(lower, upper, logpdf, *params, order=3, mean_val=mean_val)
    return third_central / (pt.sqrt(variance) ** 3)


def continuous_kurtosis(lower, upper, logpdf, *params):
    """Compute excess kurtosis for continuous distributions."""
    mean_val = continuous_moment(lower, upper, logpdf, *params, order=1)
    variance = continuous_moment(lower, upper, logpdf, *params, order=2, mean_val=mean_val)
    fourth_central = continuous_moment(lower, upper, logpdf, *params, order=4, mean_val=mean_val)
    return fourth_central / (variance**2) - 3


def from_tau(tau):
    """Convert precision (tau) to standard deviation (sigma)."""
    sigma = 1 / pt.sqrt(tau)
    return sigma


def to_tau(sigma):
    """Convert standard deviation (sigma) to precision (tau)."""
    tau = pt.power(sigma, -2)
    return tau


def von_mises_cdf(x, mu, kappa):
    """Approximate VonMises CDF with vectorized series."""
    x_centered = x - mu
    ix = pt.round(x_centered / (2 * pt.pi))
    x_wrapped = x_centered - ix * 2 * pt.pi

    CK = 50.0
    p = pt.cast(pt.clip(pt.round(1 + 28.0 + 0.5 * kappa - 100.0 / (kappa + 5.0)), 5, 50), "int32")

    use_series = pt.lt(kappa, CK)
    # arange requires a scalar stop; the per-column mask below drops the
    # extra terms for columns with a smaller p
    n_values = pt.arange(1, pt.max(p) + 1, dtype="float64")

    kappa_flat = pt.atleast_1d(kappa)  # (K,)
    n_expanded = n_values[:, None, None]  # (n, 1, 1)
    kappa_expanded = kappa_flat[None, None, :]  # (1, 1, K)
    x_expanded = pt.atleast_1d(x_wrapped)[None]  # (1, N) / (1, N, P) / (1, 1)

    bessel_ratio = (
        pt.ive(n_expanded, kappa_expanded) / pt.ive(0.0, kappa_flat)[None, None, :]
    )  # (n, 1, K)
    terms = bessel_ratio * pt.sin(n_expanded * x_expanded) / n_expanded  # (n, 1, N[, P], K)

    mask = n_values[:, None, None] <= pt.atleast_1d(p)[None, None, :]  # (n, 1, K)
    masked_terms = pt.switch(mask, terms, 0.0)
    series_sum = pt.sum(masked_terms, axis=0)  # (1, N[, P], K)

    cdf_series = pt.clip(
        0.5 + pt.atleast_1d(x_wrapped) / (2 * pt.pi) + series_sum / pt.pi, 0.0, 1.0
    )  # (1, N[, P], K)

    # Normal approximation
    b = pt.sqrt(2 / pt.pi) / pt.ive(0.0, kappa_flat)  # (K,)
    z = b * pt.sin(pt.atleast_1d(x_wrapped) / 2.0)  # (N[, P], K)
    cdf_norm = 0.5 * (1.0 + pt.erf(z / 2**0.5))[None]  # (1, N[, P], K)

    result = pt.switch(use_series, cdf_series, cdf_norm)
    result = result + ix
    return pt.squeeze(result)


def continuous_mode(lower, upper, logpdf, *params, n_points=200):
    """
    Find the mode of a continuous distribution by grid search over logpdf.

    Parameters
    ----------
    lower : float or tensor
        Lower bound of the search region (can be batched)
    upper : float or tensor
        Upper bound of the search region (can be batched)
    logpdf : function
        Log probability density function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to logpdf
    n_points : int
        Number of grid points for the search

    Returns
    -------
    mode : tensor
        The x value that maximizes the PDF
    """
    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    x_vals = pt.linspace(lower, upper, n_points)

    # When bounds are batched, linspace returns (n_points, batch_dims...) which
    # already broadcasts correctly with params. Only reshape for scalar bounds.
    if x_vals.ndim == 1 and broadcast_shape.ndim > 0:
        x_broadcast = x_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)
    else:
        x_broadcast = x_vals
    logpdf_vals = logpdf(x_broadcast, *params)

    max_idx = pt.argmax(logpdf_vals, axis=0)

    # For batched bounds, each column has different x values, so we need
    # advanced indexing to select the right x for each batch element
    if x_vals.ndim > 1:
        batch_indices = pt.arange(x_vals.shape[1])
        result = x_vals[max_idx, batch_indices]
    else:
        result = x_vals[max_idx]

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def zi_mode(base_mode, logpdf, *params):
    """
    Compute mode for zero-inflated distributions.

    Compares probability at x=0 vs probability at the base distribution's mode,
    and returns whichever has higher probability.

    Parameters
    ----------
    base_mode : tensor
        The mode of the underlying (non-zero-inflated) distribution
    logpdf : function
        Log probability function of the ZI distribution that takes (x, *params)
    *params : tensor variables
        Parameters to pass to logpdf (typically psi, plus base distribution params)

    Returns
    -------
    tensor
        The mode value (either 0 or base_mode)
    """
    return pt.switch(logpdf(0, *params) >= logpdf(base_mode, *params), 0, base_mode)


def ncx2_cdf(x, df, nc):
    """
    Compute the CDF of the noncentral chi-squared distribution.

    Uses a hybrid approach:
    - For small nc: Poisson-weighted series starting from j=0
    - For large nc: Normal approximation (ncx2 approaches normal for large nc)

    The series uses enough terms to cover the significant part of the Poisson
    distribution, extending well beyond the mode at half_nc.

    Parameters
    ----------
    x : tensor
        Value at which to evaluate the CDF
    df : float
        Degrees of freedom
    nc : tensor
        Non-centrality parameter

    Returns
    -------
    tensor
        CDF value
    """
    x = pt.as_tensor_variable(x)
    nc = pt.as_tensor_variable(nc)
    half_nc = nc / 2.0
    half_df = df / 2.0
    half_x = x / 2.0

    # Use enough terms to cover mode + several standard deviations
    # Poisson std = sqrt(lambda), so we need ~mode + 10*sqrt(mode) terms
    # For nc up to 1000, half_nc up to 500, we need ~500 + 10*22 = 720 terms
    max_terms = 800

    j = pt.arange(max_terms, dtype="float64")

    # Broadcast j over both x and nc dimensions
    bc_shape = pt.broadcast_arrays(x, nc)[0]
    j_bc = j.reshape((-1,) + (1,) * bc_shape.ndim)

    df_bc = half_df + j_bc

    # Compute log-weights: log(Poisson(j; half_nc))
    # log_w = -half_nc + j*log(half_nc) - log(j!)
    # Handle half_nc=0 case: all weight on j=0
    log_half_nc = pt.switch(pt.gt(half_nc, 0), pt.log(half_nc), -pt.inf)
    log_weights = -half_nc + j_bc * log_half_nc - pt.gammaln(j_bc + 1)

    # Exp with underflow protection
    weights_bc = pt.exp(pt.clip(log_weights, -700, 700))

    chi2_cdfs = pt.gammainc(df_bc, half_x)
    series_result = pt.sum(weights_bc * chi2_cdfs, axis=0)

    # Normal approximation for large nc
    # ncx2(df, nc) ~ N(df + nc, 2*(df + 2*nc)) for large nc
    mean_approx = df + nc
    std_approx = pt.sqrt(2 * (df + 2 * nc))
    normal_result = 0.5 * (1 + pt.erf((x - mean_approx) / (std_approx * SQRT2)))

    # Use series for nc < 1000, normal approximation otherwise
    return pt.switch(pt.lt(nc, 1000), series_result, normal_result)


def marcum_q1_complement(a, b):
    """
    Compute 1 - Q_1(a, b) where Q_1 is the Marcum Q-function.

    Uses the relationship with the noncentral chi-squared distribution:
    1 - Q_1(a, b) = CDF of ncx2(df=2, nc=a^2) evaluated at b^2

    Parameters
    ----------
    a : tensor
        First parameter
    b : tensor
        Second parameter

    Returns
    -------
    tensor
        1 - Q_1(a, b)
    """
    nc = a**2
    x = b**2

    result = ncx2_cdf(x, 2.0, nc)

    return pt.switch(pt.le(b, 0), 0.0, result)
