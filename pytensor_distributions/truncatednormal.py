import pytensor.tensor as pt

from pytensor_distributions import normal as Normal
from pytensor_distributions.helper import logdiffexp, ppf_bounds_cont
from pytensor_distributions.lmoments import _lmoments


def _phi(z):
    return 0.5 * (1 + pt.erf(z / pt.sqrt(2.0)))


def _Phi_inv(p):
    return pt.sqrt(2.0) * pt.erfinv(2 * p - 1)


def _log_phi(z):
    return -0.5 * z**2 - pt.log(pt.sqrt(2 * pt.pi))


def _alpha_beta(mu, sigma, lower, upper):
    alpha_raw = (lower - mu) / sigma
    beta_raw = (upper - mu) / sigma

    alpha = pt.switch(pt.isinf(alpha_raw) & (alpha_raw < 0), -100.0, alpha_raw)
    beta = pt.switch(pt.isinf(beta_raw) & (beta_raw > 0), 100.0, beta_raw)
    return alpha, beta


def _log_erfc(x):
    # log(erfc(x)) = log(erfcx(x) * exp(-x^2)) = log(erfcx(x)) - x^2
    # Safe for large positive x.
    return pt.log(pt.erfcx(x)) - x**2


def _log_Z(alpha, beta):
    """Compute log(Phi(beta) - Phi(alpha)) robustly."""
    a = alpha / pt.sqrt(2.0)
    b = beta / pt.sqrt(2.0)
    log_half = pt.log(0.5)

    # Case 1: alpha > 0 (implies beta > 0). Upper tail.
    # Z = 0.5 * (erfc(a) - erfc(b)). a < b.
    safe_a_pos = pt.switch(pt.gt(alpha, 0), a, 0.0)
    safe_b_pos = pt.switch(pt.gt(alpha, 0), b, 0.0)
    res_pos = log_half + logdiffexp(_log_erfc(safe_a_pos), _log_erfc(safe_b_pos))

    # Case 2: beta < 0 (implies alpha < 0). Lower tail.
    # Z = 0.5 * (erfc(-b) - erfc(-a)). -b < -a.
    safe_neg_b = pt.switch(pt.lt(beta, 0), -b, 0.0)
    safe_neg_a = pt.switch(pt.lt(beta, 0), -a, 0.0)
    res_neg = log_half + logdiffexp(_log_erfc(safe_neg_b), _log_erfc(safe_neg_a))

    # Case 3: Straddles 0. alpha <= 0 <= beta.
    # Z = 0.5 * (erf(b) - erf(a)).
    # erf(b) >= 0, erf(a) <= 0. Difference is sum of magnitudes.
    res_straddle = log_half + pt.log(pt.erf(b) - pt.erf(a))

    return pt.switch(
        pt.gt(alpha, 0),
        res_pos,
        pt.switch(pt.lt(beta, 0), res_neg, res_straddle),
    )


def mean(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    log_Z = _log_Z(alpha, beta)

    log_phi_a = _log_phi(alpha)
    log_phi_b = _log_phi(beta)

    term = pt.exp(log_phi_a - log_Z) - pt.exp(log_phi_b - log_Z)
    return mu + sigma * term


def mode(mu, sigma, lower, upper):
    return pt.clip(mu, lower, upper)


def median(mu, sigma, lower, upper):
    return ppf(0.5, mu, sigma, lower, upper)


def var(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    log_Z = _log_Z(alpha, beta)

    log_phi_a = _log_phi(alpha)
    log_phi_b = _log_phi(beta)

    term_a = pt.exp(log_phi_a - log_Z)
    term_b = pt.exp(log_phi_b - log_Z)

    term1 = alpha * term_a - beta * term_b
    term2 = pt.sqr(term_a - term_b)

    return sigma**2 * (1 + term1 - term2)


def std(mu, sigma, lower, upper):
    return pt.sqrt(var(mu, sigma, lower, upper))


def _raw_moments(alpha, beta):
    """Raw moments m1, m2, m3, m4 for standard truncated normal."""
    log_Z = _log_Z(alpha, beta)
    log_phi_a = _log_phi(alpha)
    log_phi_b = _log_phi(beta)

    phi_a_Z = pt.exp(log_phi_a - log_Z)
    phi_b_Z = pt.exp(log_phi_b - log_Z)

    m1 = phi_a_Z - phi_b_Z
    m2 = 1 + alpha * phi_a_Z - beta * phi_b_Z
    m3 = 2 * m1 + alpha**2 * phi_a_Z - beta**2 * phi_b_Z
    m4 = 3 * m2 + alpha**3 * phi_a_Z - beta**3 * phi_b_Z

    return m1, m2, m3, m4


def skewness(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    m1, m2, m3, _ = _raw_moments(alpha, beta)

    mu2 = m2 - m1**2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1**3

    return mu3 / pt.pow(mu2, 1.5)


def kurtosis(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    m1, m2, m3, m4 = _raw_moments(alpha, beta)

    mu2 = m2 - m1**2
    mu4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4

    return mu4 / mu2**2 - 3


def lmoment1(mu, sigma, lower, upper):
    return mean(mu, sigma, lower, upper)


def lmoment2(mu, sigma, lower, upper):
    return _lmoments(ppf, mu, sigma, lower, upper, r=2)


def lmoment3(mu, sigma, lower, upper):
    return _lmoments(ppf, mu, sigma, lower, upper, r=3)


def lmoment4(mu, sigma, lower, upper):
    return _lmoments(ppf, mu, sigma, lower, upper, r=4)


def entropy(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    log_Z = _log_Z(alpha, beta)

    log_phi_a = _log_phi(alpha)
    log_phi_b = _log_phi(beta)

    term_a = pt.exp(log_phi_a - log_Z)
    term_b = pt.exp(log_phi_b - log_Z)

    return (
        pt.log(pt.sqrt(2 * pt.pi * pt.e) * sigma) + log_Z + 0.5 * (alpha * term_a - beta * term_b)
    )


def pdf(x, mu, sigma, lower, upper):
    return pt.exp(logpdf(x, mu, sigma, lower, upper))


def logpdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    log_Z = _log_Z(alpha, beta)

    base_logpdf = Normal.logpdf(x, mu, sigma)
    result = base_logpdf - log_Z

    return pt.switch(
        pt.or_(pt.lt(x, lower), pt.gt(x, upper)),
        -pt.inf,
        result,
    )


def cdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    xi = (x - mu) / sigma

    return pt.switch(
        pt.le(x, lower),
        0.0,
        pt.switch(
            pt.ge(x, upper),
            1.0,
            pt.switch(
                pt.lt(xi, alpha),
                0.0,
                pt.switch(
                    pt.gt(xi, beta),
                    1.0,
                    pt.exp(_log_Z(alpha, xi) - _log_Z(alpha, beta)),
                ),
            ),
        ),
    )


def logcdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    xi = (x - mu) / sigma

    log_num = _log_Z(alpha, xi)
    log_den = _log_Z(alpha, beta)

    result = log_num - log_den

    return pt.switch(
        pt.le(x, lower),
        -pt.inf,
        pt.switch(pt.ge(x, upper), 0.0, result),
    )


def sf(x, mu, sigma, lower, upper):
    return 1.0 - cdf(x, mu, sigma, lower, upper)


def logsf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    xi = (x - mu) / sigma

    log_num = _log_Z(xi, beta)
    log_den = _log_Z(alpha, beta)

    result = log_num - log_den

    return pt.switch(
        pt.le(x, lower),
        0.0,
        pt.switch(pt.ge(x, upper), -pt.inf, result),
    )


def ppf(q, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)

    # Use survival-based computation when alpha > 0 (upper tail) for numerical stability
    def ppf_standard(q, alpha, beta):
        Z = _phi(beta) - _phi(alpha)
        return _Phi_inv(q * Z + _phi(alpha))

    def ppf_survival(q, alpha, beta):
        sb = 0.5 * pt.erfc(beta / pt.sqrt(2.0))
        sa = 0.5 * pt.erfc(alpha / pt.sqrt(2.0))
        term = q * sb + (1 - q) * sa
        return pt.sqrt(2.0) * pt.erfcinv(2 * term)

    result_standard = ppf_standard(q, alpha, beta)
    result_survival = ppf_survival(q, alpha, beta)

    result = pt.switch(pt.gt(alpha, 0), result_survival, result_standard)
    result = mu + sigma * result

    return ppf_bounds_cont(result, q, lower, upper)


def isf(q, mu, sigma, lower, upper):
    return ppf(1.0 - q, mu, sigma, lower, upper)


def rvs(mu, sigma, lower, upper, size=None, random_state=None):
    u = pt.random.uniform(0, 1, size=size, rng=random_state, return_next_rng=True)[1]
    return ppf(u, mu, sigma, lower, upper)
