"""Test Zero-Inflated Binomial distribution using empirical validation."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import kurtosis, skew

from pytensor_distributions import zi_binomial as ZIBinomial
from tests.helper_scipy import run_lmoments_test


def make_params(psi, n, p):
    """Create PyTensor constant parameters for ZI-Binomial."""
    psi_param = pt.constant(psi, dtype="float64")
    n_param = pt.constant(n, dtype="int64")
    p_param = pt.constant(p, dtype="float64")
    return (psi_param, n_param, p_param)


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 10, 0.3),  # psi, n, p - moderate zero inflation
        (0.5, 20, 0.5),  # high zero inflation, symmetric
        (0.9, 5, 0.2),  # low zero inflation, low success prob
        (0.3, 15, 0.7),  # high zero inflation, high success prob
        (0.8, 8, 0.4),  # low zero inflation, moderate
    ],
)
def test_zi_binomial_moments_vs_samples(params):
    """Test ZI Binomial moments against Monte Carlo samples."""
    psi, n, p = params
    p_params = make_params(psi, n, p)
    sample_size = 500_000

    # Generate samples
    rng_p = pt.random.default_rng(42)
    rvs = ZIBinomial.rvs(*p_params, size=sample_size, random_state=rng_p).eval()

    # Test mean
    theoretical_mean = ZIBinomial.mean(*p_params).eval()
    assert_allclose(theoretical_mean, rvs.mean(), rtol=1e-2, atol=1e-4)

    # Test variance
    theoretical_var = ZIBinomial.var(*p_params).eval()
    assert_allclose(theoretical_var, rvs.var(), rtol=1e-2, atol=1e-4)

    # Test skewness
    theoretical_skewness = ZIBinomial.skewness(*p_params).eval()
    assert_allclose(theoretical_skewness, skew(rvs), rtol=2e-1, atol=1e-2)

    # Test kurtosis
    theoretical_kurtosis = ZIBinomial.kurtosis(*p_params).eval()
    assert_allclose(theoretical_kurtosis, kurtosis(rvs), rtol=3e-1, atol=1e-2)


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 10, 0.3),
        (0.5, 20, 0.5),
    ],
)
def test_zi_binomial_pmf_properties(params):
    """Test that PMF sums to 1 and has correct zero-inflation structure."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    # PMF should sum to 1 over the support [0, n]
    x_vals = pt.arange(0, n + 1)
    pmf_sum = pt.sum(ZIBinomial.pdf(x_vals, *p_params)).eval()
    assert_allclose(pmf_sum, 1.0, rtol=1e-4, err_msg="PMF does not sum to 1")

    # Check zero-inflation structure:
    # P(X=0) should be (1-psi) + psi * (1-p)^n
    expected_p0 = (1 - psi) + psi * np.power(1 - p, n)
    actual_p0 = ZIBinomial.pdf(0, *p_params).eval()
    assert_allclose(actual_p0, expected_p0, rtol=1e-6, err_msg="P(X=0) is incorrect")

    # For x > 0, P(X=x) should be psi * Binomial.pmf(x, n, p)
    from scipy import stats

    for x in [1, 2, min(5, n)]:
        expected = psi * stats.binom.pmf(x, n, p)
        actual = ZIBinomial.pdf(x, *p_params).eval()
        assert_allclose(actual, expected, rtol=1e-6, err_msg=f"P(X={x}) is incorrect")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 10, 0.3),
        (0.5, 20, 0.5),
    ],
)
def test_zi_binomial_cdf_sf_complement(params):
    """Test that CDF + SF = 1."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    x_vals = np.array([0, 1, 2, 5, n // 2, n])
    cdf_vals = ZIBinomial.cdf(x_vals, *p_params).eval()
    sf_vals = ZIBinomial.sf(x_vals, *p_params).eval()

    assert_allclose(cdf_vals + sf_vals, 1.0, rtol=1e-6, err_msg="CDF + SF != 1")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 10, 0.3),
        (0.5, 20, 0.5),
    ],
)
def test_zi_binomial_ppf_cdf_inverse(params):
    """Test that PPF is the inverse of CDF."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    # For discrete distributions, CDF(PPF(q)) >= q
    q_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    ppf_vals = ZIBinomial.ppf(q_vals, *p_params).eval()
    cdf_at_ppf = ZIBinomial.cdf(ppf_vals, *p_params).eval()

    assert np.all(cdf_at_ppf >= q_vals - 1e-10), "PPF-CDF inverse property violated"


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 10, 0.3),
        (0.5, 20, 0.5),
    ],
)
def test_zi_binomial_mean_variance(params):
    """Test theoretical mean and variance formulas."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    # Mean = psi * n * p
    expected_mean = psi * n * p
    actual_mean = ZIBinomial.mean(*p_params).eval()
    assert_allclose(actual_mean, expected_mean, rtol=1e-6, err_msg="Mean formula is incorrect")

    # Var = psi * (n*p*(1-p) + (1-psi) * (n*p)^2)
    base_mean = n * p
    base_var = n * p * (1 - p)
    expected_var = psi * (base_var + (1 - psi) * base_mean**2)
    actual_var = ZIBinomial.var(*p_params).eval()
    assert_allclose(actual_var, expected_var, rtol=1e-6, err_msg="Variance formula is incorrect")


def test_zi_binomial_reduces_to_binomial():
    """Test that when psi=1, ZI-Binomial reduces to standard Binomial."""
    from pytensor_distributions import binomial as Binomial

    psi = 1.0
    n = 10
    p = 0.4
    p_params = make_params(psi, n, p)
    binomial_params = (pt.constant(n, dtype="int64"), pt.constant(p, dtype="float64"))

    x_vals = np.array([0, 1, 2, 3, 5, n])

    # PMF should match
    zi_pmf = ZIBinomial.pdf(x_vals, *p_params).eval()
    binomial_pmf = Binomial.pdf(x_vals, *binomial_params).eval()
    assert_allclose(zi_pmf, binomial_pmf, rtol=1e-6, err_msg="ZI-Binomial(psi=1) != Binomial")

    # CDF should match
    zi_cdf = ZIBinomial.cdf(x_vals, *p_params).eval()
    binomial_cdf = Binomial.cdf(x_vals, *binomial_params).eval()
    assert_allclose(
        zi_cdf, binomial_cdf, rtol=1e-6, err_msg="ZI-Binomial CDF(psi=1) != Binomial CDF"
    )


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 10, 0.3),
        (0.5, 20, 0.5),
    ],
)
def test_zi_binomial_bounds(params):
    """Test that PMF is zero outside support [0, n]."""
    psi, n, p = params
    p_params = make_params(psi, n, p)

    # Outside support
    outside_vals = np.array([-1, -2, n + 1, n + 2])
    pmf_outside = ZIBinomial.pdf(outside_vals, *p_params).eval()
    assert_allclose(pmf_outside, 0.0, atol=1e-10, err_msg="PMF should be 0 outside support")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 10, 0.3),
        (0.9, 20, 0.5),
    ],
)
def test_zi_binomial_lmoments(params):
    psi, n, p = params
    p_params = make_params(psi, n, p)
    run_lmoments_test(p_dist=ZIBinomial, p_params=p_params, name="zi_binomial")
