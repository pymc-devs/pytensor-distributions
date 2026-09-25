import math

import pytensor.tensor as pt
from pytensor import scan
from pytensor.scan.utils import until

from pytensor_distributions.helper import (
    isf_bounds_cont,
    isf_bounds_disc,
    ppf_bounds_cont,
    ppf_bounds_disc,
)


def find_ppf(q, x0, lower, upper, cdf_func, pdf_func, *params, max_iter=100, tol=1e-8):
    x0 = x0 + pt.zeros_like(q)

    def step(x_prev):
        x_prev_squeezed = pt.squeeze(x_prev)

        cdf_val = cdf_func(x_prev_squeezed, *params)
        f_x = pt.maximum(pdf_func(x_prev_squeezed, *params), 1e-300)
        delta = (cdf_val - q) / f_x

        max_step = pt.maximum(pt.abs(x_prev_squeezed), 1.0)
        delta = pt.clip(delta, -max_step, max_step)
        x_new = x_prev_squeezed - delta

        converged = pt.abs(x_new - x_prev_squeezed) < tol
        x_new = pt.switch(converged, x_prev_squeezed, x_new)

        all_converged = pt.all(converged)
        return pt.shape_padleft(x_new), until(all_converged)

    x_seq = scan(fn=step, outputs_info=pt.shape_padleft(x0), n_steps=max_iter, return_updates=False)

    return ppf_bounds_cont(x_seq[-1].squeeze(), q, lower, upper)


def find_isf(q, x0, lower, upper, sf_func, pdf_func, *params, max_iter=100, tol=1e-8):
    x0 = x0 + pt.zeros_like(q)

    def step(x_prev):
        x_prev_squeezed = pt.squeeze(x_prev)

        sf_val = sf_func(x_prev_squeezed, *params)
        f_x = pt.maximum(pdf_func(x_prev_squeezed, *params), 1e-300)
        delta = (q - sf_val) / f_x

        max_step = pt.maximum(pt.abs(x_prev_squeezed), 1.0)
        delta = pt.clip(delta, -max_step, max_step)
        x_new = x_prev_squeezed - delta

        converged = pt.abs(x_new - x_prev_squeezed) < tol
        x_new = pt.switch(converged, x_prev_squeezed, x_new)

        all_converged = pt.all(converged)
        return pt.shape_padleft(x_new), until(all_converged)

    x_seq = scan(fn=step, outputs_info=pt.shape_padleft(x0), n_steps=max_iter, return_updates=False)

    return isf_bounds_cont(x_seq[-1].squeeze(), q, lower, upper)


def _is_scalar_param(param):
    """Check if a parameter is a scalar (0-dimensional) at graph-build time."""
    if hasattr(param, "ndim"):
        return param.ndim == 0
    # For Python scalars
    import numpy as np

    return np.ndim(param) == 0


def _should_use_bisection(lower, upper, params, max_direct_search_size=10_000):
    """Compile-time check to select PPF algorithm for discrete distributions.

    This function inspects bounds at graph-construction time to choose between:
    - Direct search: Fast for narrow bounded support (e.g., BetaBinomial, Binomial)
    - Bisection: Required for unbounded or wide support (e.g., Poisson, NegativeBinomial)

    The check happens at Python level during graph construction, not during
    PyTensor execution. This is intentional: a fully symbolic approach using
    pt.switch would evaluate both branches, causing performance issues.

    Parameters
    ----------
    lower : int, float, or PyTensor constant
        Lower bound of the distribution support
    upper : int, float, or PyTensor constant
        Upper bound of the distribution support
    params : tuple
        Distribution parameters - if any are non-scalar, bisection is required
        to handle broadcasting correctly.
    max_direct_search_size : int, default 10_000
        Maximum range size for direct search. Larger ranges use bisection.

    Returns
    -------
    bool
        True if bisection should be used, False for direct search.
    """
    # Check if any parameter is non-scalar (array) - direct search doesn't
    # handle broadcasting correctly, so fall back to bisection
    for param in params:
        if not _is_scalar_param(param):
            return True

    try:
        # Extract constant values at graph-build time
        if hasattr(lower, "data"):
            lower_val = float(lower.data)
        else:
            lower_val = float(lower)

        if hasattr(upper, "data"):
            upper_val = float(upper.data)
        else:
            upper_val = float(upper)
    except (TypeError, ValueError):
        # Symbolic (non-constant) bounds - use bisection as safe default
        return True

    # Check for infinite bounds
    if not (math.isfinite(lower_val) and math.isfinite(upper_val)):
        return True

    # Check if range exceeds threshold
    return (upper_val - lower_val) > max_direct_search_size


def find_ppf_discrete(q, x0, lower, upper, cdf_func, pmf_func, *params, max_iter=100, tol=1e-7):
    """Find PPF for discrete distributions."""
    x0 = pt.floor(x0) + pt.zeros_like(q)

    def step(x_prev):
        x_prev_squeezed = pt.squeeze(x_prev)
        x_int = pt.floor(x_prev_squeezed)

        cdf_val = cdf_func(x_int, *params)
        cdf_val_minus = cdf_func(x_int - 1, *params)
        found = (cdf_val >= q * (1 - tol)) & (cdf_val_minus < q * (1 + tol))
        pmf_val = pt.maximum(pmf_func(x_int, *params), 1e-300)
        delta = (cdf_val - q) / pmf_val

        delta_discrete = pt.switch(pt.abs(delta) < 0.5, pt.sign(cdf_val - q), pt.floor(delta))

        x_new = x_int - delta_discrete

        x_new = pt.clip(x_new, lower, upper)
        x_new = pt.switch(found, x_int, x_new)
        all_converged = pt.all(found)

        return pt.shape_padleft(x_new), until(all_converged)

    x_seq = scan(fn=step, outputs_info=pt.shape_padleft(x0), n_steps=max_iter, return_updates=False)

    return ppf_bounds_disc(x_seq[-1].squeeze(), q, lower, upper)


def find_isf_discrete(q, x0, lower, upper, sf_func, pmf_func, *params, max_iter=200, tol=1e-7):
    """Find ISF for discrete distributions."""
    x0 = pt.floor(x0) + pt.zeros_like(q)
    lo = x0
    hi = x0
    w = pt.ones_like(x0)

    def step(lo_s, hi_s, w_s):
        lo = pt.squeeze(lo_s)
        hi = pt.squeeze(hi_s)
        w = pt.squeeze(w_s)
        s_lo = sf_func(lo, *params)
        s_hi = sf_func(hi, *params)

        lo_ok = pt.or_(pt.lt(lo, lower), pt.gt(s_lo, q * (1 - tol)))
        hi_ok = pt.le(s_hi, q * (1 + tol))
        bracketed = pt.and_(pt.and_(lo_ok, hi_ok), pt.gt(hi, lo))

        w_new = pt.where(bracketed, w, w * 2)
        lo_new = pt.where(pt.or_(~lo_ok, ~pt.gt(hi, lo)), pt.maximum(lower - 1, lo - w), lo)
        hi_new = pt.where(
            pt.and_(lo_ok, pt.or_(~hi_ok, pt.le(hi, lo))),
            pt.minimum(upper, hi + w),
            hi,
        )
        lo_exp = pt.where(~lo_ok, lo_new, lo)
        hi_exp = pt.where(~lo_ok, hi, hi_new)

        mid = pt.floor((lo_exp + hi_exp) / 2)
        s_mid = sf_func(mid, *params)
        lo_bis = pt.where(pt.gt(s_mid, q * (1 + tol)), mid, lo_exp)
        hi_bis = pt.where(pt.le(s_mid, q * (1 + tol)), mid, hi_exp)

        use_bisect = bracketed
        lo_next = pt.where(use_bisect, lo_bis, lo_exp)
        hi_next = pt.where(use_bisect, hi_bis, hi_exp)
        w_next = pt.where(use_bisect, w, w_new)

        converged = pt.and_(bracketed, pt.le(hi_next - lo_next, 1))
        return (
            pt.shape_padleft(lo_next),
            pt.shape_padleft(hi_next),
            pt.shape_padleft(w_next),
        ), until(pt.all(converged))

    _, hi_f, _ = scan(
        fn=step,
        outputs_info=(pt.shape_padleft(lo), pt.shape_padleft(hi), pt.shape_padleft(w)),
        n_steps=max_iter,
        return_updates=False,
    )
    return isf_bounds_disc(hi_f[-1].squeeze(), q, lower, upper)
