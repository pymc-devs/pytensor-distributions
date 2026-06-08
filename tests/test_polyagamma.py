"""Test PolyaGamma distribution against empirical samples."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import kurtosis, skew

from pytensor_distributions import polyagamma as PolyaGamma
from tests.helper_scipy import make_params

PARAMS_LIST = [
    (1.0, 0.0),
    (2.0, 2.0),
    (0.5, 1.0),
]


@pytest.fixture(scope="module")
def samples():
    """Generate samples once for all tests (shared across parametrizations)."""
    results = {}
    for h, z in PARAMS_LIST:
        p_params = make_params(h, z, dtype="float64")
        rng_p = pt.random.default_rng(1)
        rvs = PolyaGamma.rvs(*p_params, size=10_000, random_state=rng_p).eval()
        results[(h, z)] = (p_params, rvs)
    return results


@pytest.mark.parametrize("params", PARAMS_LIST)
def test_polyagamma_moments(params, samples):
    """Theoretical moments should match empirical moments from samples."""
    p_params, rvs = samples[params]

    assert_allclose(PolyaGamma.mean(*p_params).eval(), rvs.mean(), rtol=3e-2, atol=3e-2)
    assert_allclose(PolyaGamma.var(*p_params).eval(), rvs.var(), rtol=1e-1, atol=1e-3)
    assert_allclose(PolyaGamma.std(*p_params).eval(), rvs.std(), rtol=1e-1, atol=1e-3)
    assert_allclose(PolyaGamma.skewness(*p_params).eval(), skew(rvs), rtol=3e-1, atol=1e-2)
    assert_allclose(PolyaGamma.kurtosis(*p_params).eval(), kurtosis(rvs), rtol=3e-1, atol=1e-2)


@pytest.mark.parametrize("params", PARAMS_LIST)
def test_polyagamma_cdf(params, samples):
    """CDF should match empirical CDF and be monotonic on a small grid."""
    p_params, rvs = samples[params]

    sample_x = rvs[:20]
    theoretical_cdf = PolyaGamma.cdf(sample_x, *p_params).eval()
    for i, x in enumerate(sample_x):
        empirical_cdf = np.mean(rvs <= x)
        assert_allclose(theoretical_cdf[i], empirical_cdf, rtol=1e-1, atol=1e-3)

    x_grid = np.linspace(np.percentile(rvs, 1), np.percentile(rvs, 99), 50)
    cdf_vals = PolyaGamma.cdf(x_grid, *p_params).eval()
    assert np.all(np.diff(cdf_vals) >= -1e-4), "CDF is not monotonic"


@pytest.mark.parametrize("params", PARAMS_LIST)
def test_polyagamma_cdf_bounds(params, samples):
    """CDF should be 0 at lower bound and handle out-of-support."""
    p_params, _ = samples[params]

    extended_vals = np.array([0.0, np.inf, -1.0, -2.0, np.inf, np.inf])
    expected = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    assert_allclose(PolyaGamma.cdf(extended_vals, *p_params).eval(), expected)


@pytest.mark.parametrize("params", [(1.0, 0.0)])
def test_polyagamma_ppf(params, samples):
    """PPF should match empirical quantiles. Slow due to scan-based solver."""
    p_params, rvs = samples[params]

    q = np.linspace(0.05, 0.95, 10)
    theoretical = PolyaGamma.ppf(q, *p_params).eval()
    empirical = np.quantile(rvs, q)
    assert_allclose(theoretical, empirical, rtol=1e-1, atol=5e-3)


def test_polyagamma_ppf_cdf_inverse(samples):
    """CDF(PPF(q)) should recover q. Slow due to scan-based solver."""
    p_params, _ = samples[(1.0, 0.0)]

    q = np.array([0.1, 0.5, 0.9])
    x_vals = PolyaGamma.ppf(q, *p_params).eval()
    recovered = PolyaGamma.cdf(x_vals, *p_params).eval()
    assert_allclose(recovered, q, atol=1e-4)


@pytest.mark.parametrize("params", PARAMS_LIST)
def test_polyagamma_pdf(params, samples):
    """PDF should be non-negative and integrate to ~1."""
    p_params, rvs = samples[params]

    x_grid = np.linspace(0.01, np.percentile(rvs, 99), 100)
    pdf_vals = PolyaGamma.pdf(x_grid, *p_params).eval()
    assert np.all(pdf_vals >= 0), "PDF has negative values"

    u_b = float(np.percentile(rvs, 99.9))
    result, _ = quad(lambda x: PolyaGamma.pdf(x, *p_params).eval(), 0, u_b)
    assert np.abs(result - 1) < 0.02, f"PDF integral = {result}, should be 1"


@pytest.mark.parametrize("params", PARAMS_LIST)
def test_polyagamma_pdf_cdf_consistency(params, samples):
    """Numerical derivative of CDF should approximate PDF."""
    p_params, rvs = samples[params]

    x_mid = np.linspace(np.percentile(rvs, 5), np.percentile(rvs, 95), 20)
    eps = 1e-5
    cdf_plus = PolyaGamma.cdf(x_mid + eps, *p_params).eval()
    cdf_minus = PolyaGamma.cdf(x_mid - eps, *p_params).eval()
    numerical_pdf = (cdf_plus - cdf_minus) / (2 * eps)
    pdf_vals = PolyaGamma.pdf(x_mid, *p_params).eval()

    mask = np.abs(pdf_vals) > 1e-4
    if np.any(mask):
        rel_error = np.abs(numerical_pdf[mask] - pdf_vals[mask]) / (np.abs(pdf_vals[mask]) + 1e-10)
        assert np.all(rel_error < 1e-2), (
            f"PDF doesn't match CDF derivative. Max rel error: {np.max(rel_error)}"
        )


@pytest.mark.parametrize("params", PARAMS_LIST)
def test_polyagamma_sf_complement(params, samples):
    """SF + CDF should equal 1."""
    p_params, rvs = samples[params]

    x = rvs[:20]
    cdf_vals = PolyaGamma.cdf(x, *p_params).eval()
    sf_vals = PolyaGamma.sf(x, *p_params).eval()
    assert_allclose(cdf_vals + sf_vals, 1.0, atol=1e-4)


@pytest.mark.parametrize("params", PARAMS_LIST)
def test_polyagamma_entropy(params, samples):
    """Monte Carlo entropy should match computed entropy."""
    p_params, rvs = samples[params]

    logpdf_vals = PolyaGamma.logpdf(rvs, *p_params).eval()
    logpdf_vals = logpdf_vals[np.isfinite(logpdf_vals)]
    mc_entropy = -np.mean(logpdf_vals)
    computed_entropy = PolyaGamma.entropy(*p_params).eval()

    rel_error = np.abs(mc_entropy - computed_entropy) / (np.abs(computed_entropy) + 1e-10)
    assert rel_error < 0.1, f"Entropy mismatch. MC: {mc_entropy}, Computed: {computed_entropy}"


def test_polyagamma_mean_z_zero():
    """Mean at z=0 should equal h/4."""
    p_params = make_params(2.0, 0.0, dtype="float64")
    result = PolyaGamma.mean(*p_params).eval()
    assert_allclose(result, 0.5, rtol=1e-10)


def test_polyagamma_var_z_zero():
    """Variance at z=0 should equal h/24."""
    p_params = make_params(2.0, 0.0, dtype="float64")
    result = PolyaGamma.var(*p_params).eval()
    assert_allclose(result, 2.0 / 24, rtol=1e-10)


def test_polyagamma_pdf_positive():
    """PDF should be positive on the support."""
    p_params = make_params(1.0, 1.0, dtype="float64")
    x = np.linspace(0.01, 2.0, 100)
    pdf_vals = PolyaGamma.pdf(x, *p_params).eval()
    assert np.all(pdf_vals > 0)


def test_polyagamma_pdf_zero_outside():
    """PDF should be zero for x <= 0."""
    p_params = make_params(1.0, 1.0, dtype="float64")
    assert_allclose(PolyaGamma.pdf(-1.0, *p_params).eval(), 0.0)
    assert_allclose(PolyaGamma.pdf(0.0, *p_params).eval(), 0.0)


def test_polyagamma_logpdf_neginf_outside():
    """Logpdf should be -inf for x <= 0."""
    p_params = make_params(1.0, 1.0, dtype="float64")
    assert PolyaGamma.logpdf(-1.0, *p_params).eval() == -np.inf
    assert PolyaGamma.logpdf(0.0, *p_params).eval() == -np.inf
