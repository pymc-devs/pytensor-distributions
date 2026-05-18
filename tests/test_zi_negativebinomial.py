"""Test Zero-Inflated Negative Binomial distribution using empirical validation."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import kurtosis, skew

from pytensor_distributions import zi_negativebinomial as ZINegativeBinomial
from tests.helper_scipy import run_lmoments_test


def make_params(*values, dtype="float64"):
    """Create PyTensor constant parameters."""
    return tuple(pt.constant(v, dtype=dtype) for v in values)


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 5.0, 0.4),  # psi, n, p - moderate zero inflation
        (0.5, 10.0, 0.3),  # high zero inflation
        (0.9, 2.0, 0.5),  # low zero inflation
        (0.6, 3.0, 0.6),  # moderate params
        (0.8, 8.0, 0.2),  # low zero inflation, high dispersion
    ],
)
def test_zi_negativebinomial_moments_vs_samples(params):
    """Test ZI Negative Binomial moments against Monte Carlo samples."""
    psi, n, p = params
    p_params = make_params(psi, n, p)
    sample_size = 500_000

    # Generate samples
    rng_p = pt.random.default_rng(42)
    rvs = ZINegativeBinomial.rvs(*p_params, size=sample_size, random_state=rng_p).eval()

    # Test mean
    theoretical_mean = ZINegativeBinomial.mean(*p_params).eval()
    assert_allclose(theoretical_mean, rvs.mean(), rtol=1e-2, atol=1e-4)

    # Test variance
    theoretical_var = ZINegativeBinomial.var(*p_params).eval()
    assert_allclose(theoretical_var, rvs.var(), rtol=1e-2, atol=1e-4)

    # Test skewness
    theoretical_skewness = ZINegativeBinomial.skewness(*p_params).eval()
    assert_allclose(theoretical_skewness, skew(rvs), rtol=2e-1, atol=1e-2)

    # Test kurtosis
    theoretical_kurtosis = ZINegativeBinomial.kurtosis(*p_params).eval()
    assert_allclose(theoretical_kurtosis, kurtosis(rvs), rtol=3e-1, atol=1e-2)


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 5.0, 0.4),
        (0.5, 10.0, 0.3),
    ],
)
def test_zi_negativebinomial_pmf_properties(params):
    """Test that PMF sums to 1 and has correct zero-inflation structure."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    # PMF should sum to approximately 1 over reasonable range
    x_vals = pt.arange(0, 100)
    pmf_sum = pt.sum(ZINegativeBinomial.pdf(x_vals, *p_params)).eval()
    assert_allclose(pmf_sum, 1.0, rtol=1e-3, err_msg="PMF does not sum to 1")

    # Check zero-inflation structure:
    # P(X=0) should be (1-psi) + psi * p^n
    expected_p0 = (1 - psi) + psi * np.power(p, n)
    actual_p0 = ZINegativeBinomial.pdf(0, *p_params).eval()
    assert_allclose(actual_p0, expected_p0, rtol=1e-6, err_msg="P(X=0) is incorrect")

    # For x > 0, P(X=x) should be psi * NegBinom.pmf(x, n, p)
    from scipy import stats

    for x in [1, 2, 5, 10]:
        expected = psi * stats.nbinom.pmf(x, n, p)
        actual = ZINegativeBinomial.pdf(x, *p_params).eval()
        assert_allclose(actual, expected, rtol=1e-6, err_msg=f"P(X={x}) is incorrect")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 5.0, 0.4),
        (0.5, 10.0, 0.3),
    ],
)
def test_zi_negativebinomial_cdf_sf_complement(params):
    """Test that CDF + SF = 1."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    x_vals = np.array([0, 1, 2, 5, 10, 20])
    cdf_vals = ZINegativeBinomial.cdf(x_vals, *p_params).eval()
    sf_vals = ZINegativeBinomial.sf(x_vals, *p_params).eval()

    assert_allclose(cdf_vals + sf_vals, 1.0, rtol=1e-6, err_msg="CDF + SF != 1")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 5.0, 0.4),
        (0.5, 10.0, 0.3),
    ],
)
def test_zi_negativebinomial_ppf_cdf_inverse(params):
    """Test that PPF is the inverse of CDF."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    # For discrete distributions, CDF(PPF(q)) >= q
    q_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    ppf_vals = ZINegativeBinomial.ppf(q_vals, *p_params).eval()
    cdf_at_ppf = ZINegativeBinomial.cdf(ppf_vals, *p_params).eval()

    assert np.all(cdf_at_ppf >= q_vals - 1e-10), "PPF-CDF inverse property violated"


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 5.0, 0.4),
        (0.5, 10.0, 0.3),
    ],
)
def test_zi_negativebinomial_mean_variance(params):
    """Test theoretical mean and variance formulas."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    # Mean = psi * n * (1-p) / p
    base_mean = n * (1 - p) / p
    expected_mean = psi * base_mean
    actual_mean = ZINegativeBinomial.mean(*p_params).eval()
    assert_allclose(actual_mean, expected_mean, rtol=1e-6, err_msg="Mean formula is incorrect")

    # Var = psi * (base_var + (1-psi) * base_mean^2)
    base_var = n * (1 - p) / (p * p)
    expected_var = psi * (base_var + (1 - psi) * base_mean**2)
    actual_var = ZINegativeBinomial.var(*p_params).eval()
    assert_allclose(actual_var, expected_var, rtol=1e-6, err_msg="Variance formula is incorrect")


def test_zi_negativebinomial_reduces_to_negativebinomial():
    """Test that when psi=1, ZI-NegBinom reduces to standard NegBinom."""
    from pytensor_distributions import negativebinomial as NegativeBinomial

    psi = 1.0
    n = 5.0
    p = 0.4
    p_params = make_params(psi, n, p)
    nb_params = make_params(n, p)

    x_vals = np.array([0, 1, 2, 3, 5, 10])

    # PMF should match
    zi_pmf = ZINegativeBinomial.pdf(x_vals, *p_params).eval()
    nb_pmf = NegativeBinomial.pdf(x_vals, *nb_params).eval()
    assert_allclose(zi_pmf, nb_pmf, rtol=1e-6, err_msg="ZI-NegBinom(psi=1) != NegBinom")

    # CDF should match
    zi_cdf = ZINegativeBinomial.cdf(x_vals, *p_params).eval()
    nb_cdf = NegativeBinomial.cdf(x_vals, *nb_params).eval()
    assert_allclose(zi_cdf, nb_cdf, rtol=1e-6, err_msg="ZI-NegBinom CDF(psi=1) != NegBinom CDF")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 5.0, 0.4),
        (0.5, 10.0, 0.3),
    ],
)
def test_zi_negativebinomial_bounds(params):
    """Test that PMF is zero for negative values."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    # Negative values should have zero probability
    negative_vals = np.array([-1, -2, -5])
    pmf_negative = ZINegativeBinomial.pdf(negative_vals, *p_params).eval()
    assert_allclose(pmf_negative, 0.0, atol=1e-10, err_msg="PMF should be 0 for negative values")


def test_zi_negativebinomial_parameterization_conversion():
    """Test the mu-alpha parameterization conversion."""
    psi = 0.7
    mu = 5.0
    alpha = 2.0

    # Convert to n, p (these are pytensor tensors)
    psi_out, n, p = ZINegativeBinomial.from_mu_alpha(psi, mu, alpha)

    # Check psi is unchanged
    assert psi_out == psi

    # Convert back
    psi_back, mu_back, alpha_back = ZINegativeBinomial.to_mu_alpha(psi, n, p)

    # Evaluate results
    mu_back_val = mu_back.eval() if hasattr(mu_back, "eval") else mu_back
    alpha_back_val = alpha_back.eval() if hasattr(alpha_back, "eval") else alpha_back

    assert_allclose(psi_back, psi, rtol=1e-6)
    assert_allclose(mu_back_val, mu, rtol=1e-6)
    assert_allclose(alpha_back_val, alpha, rtol=1e-6)


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 5.0, 0.4),
        (0.9, 10.0, 0.3),
    ],
)
def test_zi_negativebinomial_lmoments(params):
    psi, n, p = params
    p_params = make_params(psi, n, p, dtype="float64")
    run_lmoments_test(p_dist=ZINegativeBinomial, p_params=p_params, name="zi_negativebinomial")
