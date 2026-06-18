import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.special import betaln

from pytensor_distributions.genpareto import (
    _gpd_log_H,
    _gpd_log_h,
    _gpd_log_S,
    _gpd_quantile_from_excess,
    _gpd_upper_bound,
    _in_gpd_support,
    _log1p_div,
)
from pytensor_distributions.helper import continuous_mode, ppf_bounds_cont

# Extended Generalized Pareto core. Naveau et al. (2016) extended GPD with
# carrier G(v) = v ** kappa: the CDF is F = H ** kappa with H the GPD CDF, so
# kappa > 0 reshapes the lower tail while xi keeps controlling the upper tail.
# kappa = 1 recovers the plain GPD.


def logpdf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    # log g = log kappa + (kappa - 1) log H + log h. The carrier term vanishes
    # at kappa = 1; guarding it keeps the GPD reduction exact at the lower
    # endpoint (z = 0, log H = -inf), where (kappa - 1) * log H would be 0 * -inf.
    carrier = pt.switch(pt.eq(kappa, 1.0), 0.0, (kappa - 1) * _gpd_log_H(z, xi))
    logp = pt.log(kappa) + carrier + _gpd_log_h(z, sigma, xi)
    logp = pt.switch(_in_gpd_support(z, xi), logp, -np.inf)
    logp = pt.switch(pt.eq(z, np.inf), -np.inf, logp)
    return logp


def logcdf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + xi * z, 0))
    logcdf = pt.switch(above_upper, 0.0, kappa * _gpd_log_H(z, xi))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    logcdf = pt.switch(pt.eq(z, np.inf), 0.0, logcdf)
    return logcdf


def logsf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    a = _gpd_log_S(z, xi)  # log(1 - H), accurate in the tail
    log_H = pt.log1mexp(a)
    generic = pt.log1mexp(kappa * log_H)
    s = pt.exp(a)  # S_gpd
    # r = kappa * S_gpd, formed via exp(log kappa + a) and capped at 1 so the unused
    # (generic-branch) tail expression cannot overflow.
    r = pt.exp(pt.minimum(pt.log(kappa) + a, 0.0))
    # 1 - H**kappa = kappa S [1 + (s - r)/2 + (r**2 - 3 r s + 2 s**2)/6 + ...].
    # This is a Taylor expansion in *both* S_gpd and kappa*S_gpd, so it is only
    # valid where both are small; writing it in r = kappa S and s = S keeps every
    # term bounded (no kappa**k powers, which overflow for huge kappa).
    series_m1 = (s - r) / 2.0 + (r * r - 3.0 * r * s + 2.0 * s * s) / 6.0
    tail = pt.log(kappa) + a + pt.log1p(series_m1)
    # The Taylor tail runs where S_gpd and kappa*S_gpd are both below machine epsilon
    # (a < log_eps and log(kappa) + a < log_eps); the generic log1mexp(kappa log H) runs
    # otherwise. Both conditions gate the switch: for tiny kappa the second holds in the body
    # (S_gpd ~ 1) too, where the Taylor does not apply.
    log_eps = float(np.log(np.finfo(a.dtype).eps))
    logsf = pt.switch(pt.and_(a < log_eps, pt.log(kappa) + a < log_eps), tail, generic)
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + xi * z, 0))
    logsf = pt.switch(pt.or_(above_upper, pt.eq(z, np.inf)), -np.inf, logsf)
    logsf = pt.switch(z < 0, 0.0, logsf)
    return logsf


def _ext_gpd_excess_from_log_prob(log_q, kappa):
    """GPD excess ``m = -log(1 - F ** (1/kappa))`` from ``log_q = log F``.

    For the carrier ``F = H ** kappa``, the GPD CDF is ``H = exp(log_q / kappa)``
    and its survival ``1 - H``, so ``m = -log(1 - H) = -log1mexp(log_q / kappa)``
    (``pt.log1mexp(a) = log(1 - exp(a))`` for ``a <= 0``). This form stays accurate when ``H``
    rounds to ``1``: for small ``kappa`` the survival ``1 - H`` is tiny, and forming it directly
    would collapse the excess to ``0`` (and the quantile to ``mu``). Shared by
    ``ppf``, ``isf`` and ``rvs`` so the inverses agree. ``log_q`` must be ``<= 0``
    (a log-probability).
    """
    return -pt.log1mexp(log_q / kappa)


def _ext_gpd_excess_from_logit(value, kappa):
    """GPD excess ``m = -log S(x)`` from ``value = logit(F(x))``.

    The ExtGPD CDF is ``F = G ** kappa = sigmoid(value)`` (``G`` the GPD CDF), so the excess
    depends only on ``value`` and ``kappa``:
        m = -log(1 - G) = -log(1 - sigmoid(value) ** (1/kappa)).
    Evaluated stably in two branches split at a large cutoff. Writing
    a := -log G = -log(sigmoid(value)) / kappa gives m = -log1mexp(-a):
      bulk (value < cutoff): m = -log1mexp(log_F / kappa), log_F = log sigmoid(value).
      tail (value >= cutoff, the extreme upper tail where F -> 1): a held in logs as log_a,
        m = -log_a where a < eps, else -log1mexp(-a).
    The unused branch's inputs are clamped (log_F via cutoff, a via log_max) so it stays finite.
    Backs ``ppf_logit``.
    """
    finfo = np.finfo(value.dtype)
    log_max = float(np.log(finfo.max))  # a = exp(min(log_a, log_max)) stays finite
    log_eps = float(np.log(finfo.eps))  # at or below log_eps, m = -log_a
    # cutoff is the large value where log_F / kappa reaches finfo.tiny. Near the tail
    # log_F = -exp(-value), giving value = -log(tiny) - log(kappa); for kappa <= 1 it is -log(tiny).
    cutoff = np.asarray(-np.log(finfo.tiny), dtype=value.dtype) - pt.maximum(
        np.asarray(0.0, value.dtype), pt.log(kappa)
    )
    t = pt.softplus(value)
    log_F = -pt.softplus(-pt.minimum(value, cutoff))
    m_bulk = _ext_gpd_excess_from_log_prob(log_F, kappa)
    s = pt.exp(-pt.maximum(t, cutoff))  # S_ext, clamped so _log1p_div stays finite
    log_a = -t + pt.log(_log1p_div(-s)) - pt.log(kappa)
    a = pt.exp(pt.minimum(log_a, log_max))
    m_tail = pt.switch(log_a < log_eps, -log_a, -pt.log1mexp(-a))
    return pt.switch(value < cutoff, m_bulk, m_tail)


def ppf(q, mu, sigma, xi, kappa):
    q = pt.as_tensor_variable(q)
    # F = H ** kappa = q  ->  H = q ** (1/kappa); excess m = -log(1 - H), built
    # with log1mexp so a tiny 1 - H (small kappa) is not rounded away to 0.
    excess = _ext_gpd_excess_from_log_prob(pt.log(q), kappa)
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    return ppf_bounds_cont(x, q, mu, _gpd_upper_bound(mu, sigma, xi))


def cdf(x, mu, sigma, xi, kappa):
    return pt.exp(logcdf(x, mu, sigma, xi, kappa))


def pdf(x, mu, sigma, xi, kappa):
    return pt.exp(logpdf(x, mu, sigma, xi, kappa))


def sf(x, mu, sigma, xi, kappa):
    return pt.exp(logsf(x, mu, sigma, xi, kappa))


def isf(x, mu, sigma, xi, kappa):
    x = pt.as_tensor_variable(x)
    # log F = log1p(-x), accurate for tiny x; ppf(1 - x) forms 1 - x first and loses it.
    excess = _ext_gpd_excess_from_log_prob(pt.log1p(-x), kappa)
    quantile = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    return ppf_bounds_cont(quantile, x, _gpd_upper_bound(mu, sigma, xi), mu)


def ppf_logit(y, mu, sigma, xi, kappa):
    """Quantile from the logit-CDF coordinate ``y = logit(F(x)) = logcdf - logsf``.

    Equivalent to ``ppf(expit(y))`` but evaluated in log space (see
    ``_ext_gpd_excess_from_logit``), so it stays accurate when ``F`` saturates to 0 or
    1 deep in either tail. ``y`` is unconstrained (standard ``Logistic`` under the
    model), so no bounds are applied. This is the inverse for a logit-CDF
    (probability-integral) reparametrization, the stable unconstrained transform for
    sampling the extended GPD as a latent variable.
    """
    excess = _ext_gpd_excess_from_logit(y, kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)


def rvs(mu, sigma, xi, kappa, size=None, random_state=None):
    # Inverse-CDF on a carrier draw u = F; excess = -log(1 - u ** (1/kappa)).
    u = pt.random.uniform(size=size, rng=random_state, return_next_rng=True)[1]
    excess = _ext_gpd_excess_from_log_prob(pt.log(u), kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)


# Summary statistics. With the substitution x = mu + (sigma/xi)(W ** (-xi) - 1),
# W ~ Beta(1, kappa), the power moments reduce to Beta functions:
#   E[W ** (-j xi)] = kappa * B(1 - j xi, kappa),
# finite (like the plain GPD) only for j xi < 1, so each moment is guarded. The
# exponential-tail limit xi -> 0 is a 0/0 in those Beta forms, so each statistic switches
# to its closed-form digamma/polygamma limit there; ``_safe_xi`` keeps the discarded
# branch finite so its gradient is not poisoned.


def _pow_moment(j, xi, kappa):
    """``E[W ** (-j xi)] = kappa * B(1 - j xi, kappa)`` for ``W ~ Beta(1, kappa)``."""
    return pt.exp(pt.log(kappa) + betaln(1 - j * xi, kappa))


def _safe_xi(xi):
    """Nonzero stand-in for ``xi`` used only in the discarded ``xi != 0`` moment branch.

    The Beta-function forms divide by ``xi``; at ``xi = 0`` that branch is not selected,
    but ``pt.switch`` still differentiates it, so a raw ``0/0`` would make the gradient
    ``nan``. The placeholder ``0.1`` lies in ``(0, 1/4)``, keeping every ``B(1 - j xi,
    kappa)`` (``j <= 4``) and ``B(n kappa, 1 - xi)`` finite.
    """
    return pt.switch(pt.eq(xi, 0.0), 0.1, xi)


def _central_moments(sigma, xi, kappa):
    """Central moments ``E[(X - mu) ** r]`` for r = 1..4 (about mu, not the mean)."""
    c = sigma / xi
    m1, m2, m3, m4 = (_pow_moment(j, xi, kappa) for j in (1, 2, 3, 4))
    p1 = c * (m1 - 1)
    p2 = c**2 * (m2 - 2 * m1 + 1)
    p3 = c**3 * (m3 - 3 * m2 + 3 * m1 - 1)
    p4 = c**4 * (m4 - 4 * m3 + 6 * m2 - 4 * m1 + 1)
    return p1, p2, p3, p4


def mean(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    general = mu + sigma / sx * (_pow_moment(1, sx, kappa) - 1)
    limit = mu + sigma * (pt.psi(kappa + 1) + pt.euler_gamma)  # xi -> 0
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 1), value, np.inf)


def median(mu, sigma, xi, kappa):
    # F = 1/2  ->  H = (1/2) ** (1/kappa); excess m = -log(1 - H).
    excess = _ext_gpd_excess_from_log_prob(np.log(0.5), kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)


def mode(mu, sigma, xi, kappa):
    # No closed form for kappa != 1, so the interior case is a grid search:
    #   xi < -1:                 density diverges at the finite upper endpoint (h blows up);
    #   xi >= -1 and kappa <= 1: density peaks at the lower endpoint mu (H ** (kappa - 1)
    #                            blows up for kappa < 1; the kappa = 1 GPD decreases from mu);
    #   xi >= -1 and kappa > 1:  interior mode.
    # This matches GenPareto.mode at kappa = 1.
    upper = _gpd_upper_bound(mu, sigma, xi)
    interior = continuous_mode(
        mu,
        ppf(np.asarray(1.0 - 1e-3), mu, sigma, xi, kappa),
        logpdf,
        mu,
        sigma,
        xi,
        kappa,
        n_points=1000,
    )
    at_mu = pt.broadcast_arrays(mu, sigma, xi, kappa)[0]
    return pt.switch(pt.lt(xi, -1), upper, pt.switch(pt.le(kappa, 1.0), at_mu, interior))


def var(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    p1, p2, _, _ = _central_moments(sigma, sx, kappa)
    limit = sigma**2 * (pt.polygamma(1, 1.0) - pt.polygamma(1, kappa + 1))  # xi -> 0
    value = pt.switch(pt.eq(xi, 0.0), limit, p2 - p1**2)
    return pt.switch(pt.lt(xi, 0.5), value, np.inf)


def std(mu, sigma, xi, kappa):
    return pt.sqrt(var(mu, sigma, xi, kappa))


def skewness(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    p1, p2, p3, _ = _central_moments(sigma, sx, kappa)
    general = (p3 - 3 * p1 * p2 + 2 * p1**3) / (p2 - p1**2) ** 1.5
    tg = pt.polygamma(1, 1.0) - pt.polygamma(1, kappa + 1)
    limit = (pt.polygamma(2, kappa + 1) - pt.polygamma(2, 1.0)) / tg**1.5  # xi -> 0
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 1.0 / 3.0), value, np.nan)


def kurtosis(mu, sigma, xi, kappa):
    # Excess kurtosis.
    sx = _safe_xi(xi)
    p1, p2, p3, p4 = _central_moments(sigma, sx, kappa)
    variance = p2 - p1**2
    general = (p4 - 4 * p1 * p3 + 6 * p1**2 * p2 - 3 * p1**4) / variance**2 - 3
    tg = pt.polygamma(1, 1.0) - pt.polygamma(1, kappa + 1)
    limit = (pt.polygamma(3, 1.0) - pt.polygamma(3, kappa + 1)) / tg**2  # xi -> 0
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 0.25), value, np.nan)


def entropy(mu, sigma, xi, kappa):
    # -E[log g(X)] in closed form; reduces to the GPD's log(sigma) + xi + 1 at kappa = 1.
    return (
        pt.log(sigma)
        - pt.log(kappa)
        + 1
        - 1 / kappa
        + (1 + xi) * (pt.psi(kappa + 1) + pt.euler_gamma)
    )


# L-moments. With lambda_{r+1} = (sigma kappa / xi) * sum_k p*_{r,k} B(kappa (k+1), 1 - xi)
# (p* the shifted Legendre coefficients), these reduce to the GPD L-moments at kappa = 1
# and are finite for xi < 1. Like the ordinary moments they are a 0/0 at xi = 0 and switch
# to a digamma limit there.


def _lbeta(n, xi, kappa):
    """``B(n kappa, 1 - xi)``."""
    return pt.exp(betaln(n * kappa, 1 - xi))


def lmoment1(mu, sigma, xi, kappa):
    return mean(mu, sigma, xi, kappa)


def lmoment2(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    l1, l2 = _lbeta(1, sx, kappa), _lbeta(2, sx, kappa)
    general = sigma * kappa / sx * (2 * l2 - l1)
    limit = sigma * (pt.psi(2 * kappa + 1) - pt.psi(kappa + 1))  # xi -> 0
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 1), value, np.inf)


def lmoment3(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    l1, l2, l3 = (_lbeta(n, sx, kappa) for n in (1, 2, 3))
    general = (6 * l3 - 6 * l2 + l1) / (2 * l2 - l1)
    denom = pt.psi(2 * kappa + 1) - pt.psi(kappa + 1)
    limit = (2 * pt.psi(3 * kappa + 1) - 3 * pt.psi(2 * kappa + 1) + pt.psi(kappa + 1)) / denom
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 1), value, np.inf)


def lmoment4(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    l1, l2, l3, l4 = (_lbeta(n, sx, kappa) for n in (1, 2, 3, 4))
    general = (20 * l4 - 30 * l3 + 12 * l2 - l1) / (2 * l2 - l1)
    denom = pt.psi(2 * kappa + 1) - pt.psi(kappa + 1)
    limit = (
        5 * pt.psi(4 * kappa + 1)
        - 10 * pt.psi(3 * kappa + 1)
        + 6 * pt.psi(2 * kappa + 1)
        - pt.psi(kappa + 1)
    ) / denom
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 1), value, np.inf)
