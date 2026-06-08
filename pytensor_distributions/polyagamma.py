import pytensor.tensor as pt
from pytensor import scan
from pytensor.scan.utils import until

from pytensor_distributions.helper import (
    cdf_bounds,
    continuous_entropy,
    continuous_kurtosis,
    continuous_mode,
    continuous_skewness,
    ppf_bounds_cont,
)


def _lower_bound():
    return 1e-10


def _upper_bound(h, z):
    m = mean(h, z)
    s = std(h, z)
    return m + 10 * s


def _log_cosh_half(z):
    """Compute log(cosh(z/2)) in a numerically stable way."""
    abs_half_z = pt.abs(z / 2)
    return abs_half_z + pt.log1p(pt.exp(-2 * abs_half_z)) - pt.log(2.0)


def _log_pg_density_base(x, h, N=20):
    """Log density of PG(h, 0) using truncated alternating series.

    Uses the Jacobi series representation:
    f(x; h, 0) = (2^{h-1} / Gamma(h)) * sum_{n=0}^{N-1} (-1)^n
        * [Gamma(n+h) / n!] * (2n+h) / sqrt(2*pi*x^3)
        * exp(-(2n+h)^2 / (8x))

    Direct signed summation in shifted linear space avoids the pairing
    approach which fails when consecutive terms are not monotonically decreasing.
    """
    n = pt.arange(N, dtype="float64")
    n_bc = n.reshape((-1,) + (1,) * x.ndim)

    c = 2 * n_bc + h
    signs = pt.switch(pt.eq(n_bc % 2, 0), 1.0, -1.0)

    # Log of absolute value of each term (without the global prefactor)
    log_abs_term = (
        pt.gammaln(n_bc + h)
        - pt.gammaln(n_bc + 1)
        + pt.log(c)
        - 0.5 * pt.log(2 * pt.pi)
        - 1.5 * pt.log(x)
        - c**2 / (8 * x)
    )

    # Shift to prevent overflow, then sum with signs in linear space
    max_log = pt.max(log_abs_term, axis=0)
    shifted = pt.exp(log_abs_term - max_log)
    signed_sum = pt.sum(signs * shifted, axis=0)

    # Clamp to small positive value for numerical safety in far tail
    log_series = pt.log(pt.maximum(signed_sum, 1e-300)) + max_log

    # Global prefactor: (h-1)*log(2) - gammaln(h)
    log_prefactor = (h - 1) * pt.log(2.0) - pt.gammaln(h)

    return log_prefactor + log_series


def mean(h, z):
    z = pt.as_tensor_variable(z)
    small = pt.lt(pt.abs(z), 1e-6)
    safe_z = pt.switch(small, pt.ones_like(z), z)
    result_general = h / (2 * safe_z) * pt.tanh(safe_z / 2)
    result_zero = h / 4.0
    return pt.switch(small, result_zero, result_general)


def mode(h, z):
    return continuous_mode(_lower_bound(), _upper_bound(h, z), logpdf, h, z)


def median(h, z):
    return ppf(0.5, h, z)


def var(h, z):
    z = pt.as_tensor_variable(z)
    small = pt.lt(pt.abs(z), 1e-6)
    safe_z = pt.switch(small, pt.ones_like(z), z)
    result_general = h / (4 * safe_z**3) * (pt.sinh(safe_z) - safe_z) / pt.cosh(safe_z / 2) ** 2
    result_zero = h / 24.0
    return pt.switch(small, result_zero, result_general)


def std(h, z):
    return pt.sqrt(var(h, z))


def skewness(h, z):
    return continuous_skewness(_lower_bound(), _upper_bound(h, z), logpdf, h, z)


def kurtosis(h, z):
    return continuous_kurtosis(_lower_bound(), _upper_bound(h, z), logpdf, h, z)


def entropy(h, z):
    return continuous_entropy(_lower_bound(), _upper_bound(h, z), logpdf, h, z)


def logpdf(x, h, z):
    x = pt.as_tensor_variable(x)
    log_tilt = h * _log_cosh_half(z) - z**2 * x / 2
    result = log_tilt + _log_pg_density_base(x, h)
    return pt.switch(pt.le(x, 0), -pt.inf, result)


def pdf(x, h, z):
    return pt.exp(logpdf(x, h, z))


def cdf(x, h, z):
    x = pt.as_tensor_variable(x)
    n_points = 500
    lower = _lower_bound()

    t = pt.linspace(lower, x, n_points)
    pdf_vals = pdf(t, h, z)
    dx = (x - lower) / (n_points - 1)
    result = dx * (0.5 * pdf_vals[0] + pt.sum(pdf_vals[1:-1], axis=0) + 0.5 * pdf_vals[-1])

    return cdf_bounds(result, x, 0, pt.inf)


def logcdf(x, h, z):
    return pt.switch(pt.le(x, 0), -pt.inf, pt.log(cdf(x, h, z)))


def sf(x, h, z):
    return 1.0 - cdf(x, h, z)


def logsf(x, h, z):
    return pt.log1p(-cdf(x, h, z))


def ppf(q, h, z, max_iter=50, tol=1e-8):
    # Use log-normal approximation as initial guess to avoid Newton oscillation.
    # PG(h, z) is positive and right-skewed; a log-normal is a good proxy.
    m = mean(h, z)
    v = var(h, z)
    sigma_ln_sq = pt.log1p(v / m**2)
    mu_ln = pt.log(m) - sigma_ln_sq / 2
    sigma_ln = pt.sqrt(sigma_ln_sq)
    x0 = pt.exp(mu_ln + sigma_ln * pt.sqrt(2.0) * pt.erfinv(2 * q - 1))
    x0 = pt.maximum(x0, _lower_bound())

    lb = _lower_bound()

    def step(x_prev):
        x_prev_squeezed = pt.squeeze(x_prev)

        cdf_val = cdf(x_prev_squeezed, h, z)
        f_x = pt.maximum(pdf(x_prev_squeezed, h, z), 1e-10)
        delta = (cdf_val - q) / f_x

        max_step = pt.maximum(pt.abs(x_prev_squeezed), 0.5)
        delta = pt.clip(delta, -max_step, max_step)
        x_new = pt.maximum(x_prev_squeezed - delta, lb)

        converged = pt.abs(x_new - x_prev_squeezed) < tol
        x_new = pt.switch(converged, x_prev_squeezed, x_new)

        all_converged = pt.all(converged)
        return pt.shape_padleft(x_new), until(all_converged)

    x_seq = scan(fn=step, outputs_info=pt.shape_padleft(x0), n_steps=max_iter, return_updates=False)

    return ppf_bounds_cont(x_seq[-1].squeeze(), q, 0, pt.inf)


def isf(q, h, z):
    return ppf(1.0 - q, h, z)


def rvs(h, z, size=None, random_state=None):
    K = 200
    k = pt.arange(1, K + 1, dtype="float64")

    if size is None:
        gamma_size = (K,)
    elif isinstance(size, int):
        gamma_size = (K, size)
    else:
        gamma_size = (K, *size)

    gamma_draws = pt.random.gamma(
        h, scale=1.0, size=gamma_size, rng=random_state, return_next_rng=True
    )[1]

    z2_term = z**2 / (4 * pt.pi**2)
    k_bc = k.reshape((-1,) + (1,) * (gamma_draws.ndim - 1))
    denom = (k_bc - 0.5) ** 2 + z2_term

    return pt.sum(gamma_draws / denom, axis=0) / (2 * pt.pi**2)
