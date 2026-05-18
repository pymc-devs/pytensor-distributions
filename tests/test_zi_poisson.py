"""Test Zero-Inflated Poisson distribution using empirical validation."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose

from pytensor_distributions import zi_poisson as ZIPoisson
from tests.helper_empirical import run_empirical_tests
from tests.helper_scipy import run_lmoments_test


def make_params(*values, dtype="float64"):
    """Create PyTensor constant parameters."""
    return tuple(pt.constant(v, dtype=dtype) for v in values)


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 2.0),  # psi, mu - moderate zero inflation
        (0.5, 5.0),  # high zero inflation, moderate rate
        (0.9, 1.0),  # low zero inflation, low rate
        (0.3, 10.0),  # high zero inflation, high rate
        (0.8, 0.5),  # low zero inflation, low rate
    ],
)
def test_zi_poisson_empirical(params):
    """Test ZI Poisson distribution using empirical validation."""
    psi, mu = params
    p_params = make_params(psi, mu)
    support = (0, float("inf"))

    run_empirical_tests(
        p_dist=ZIPoisson,
        p_params=p_params,
        support=support,
        name="zi_poisson",
        is_discrete=True,
        sample_size=500_000,
        mean_rtol=1e-2,
        var_rtol=1e-2,
        std_rtol=1e-2,
        skewness_rtol=2e-1,
        kurtosis_rtol=3e-1,
        quantiles_rtol=1e-2,
        cdf_rtol=1e-2,
    )


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 2.0),
        (0.5, 5.0),
    ],
)
def test_zi_poisson_pmf_properties(params):
    """Test that PMF sums to 1 and has correct zero-inflation structure."""
    psi, mu = params
    p_params = make_params(psi, mu)

    # PMF should sum to 1 over reasonable range
    x_vals = pt.arange(0, 50)
    pmf_sum = pt.sum(ZIPoisson.pdf(x_vals, *p_params)).eval()
    assert_allclose(pmf_sum, 1.0, rtol=1e-4, err_msg="PMF does not sum to 1")

    # Check zero-inflation structure:
    # P(X=0) should be (1-psi) + psi * exp(-mu)
    expected_p0 = (1 - psi) + psi * np.exp(-mu)
    actual_p0 = ZIPoisson.pdf(0, *p_params).eval()
    assert_allclose(actual_p0, expected_p0, rtol=1e-6, err_msg="P(X=0) is incorrect")

    # For x > 0, P(X=x) should be psi * Poisson.pmf(x, mu)
    from scipy import stats

    for x in [1, 2, 5]:
        expected = psi * stats.poisson.pmf(x, mu)
        actual = ZIPoisson.pdf(x, *p_params).eval()
        assert_allclose(actual, expected, rtol=1e-6, err_msg=f"P(X={x}) is incorrect")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 2.0),
        (0.5, 5.0),
    ],
)
def test_zi_poisson_cdf_sf_complement(params):
    """Test that CDF + SF = 1."""
    psi, mu = params
    p_params = make_params(psi, mu)

    x_vals = np.array([0, 1, 2, 5, 10])
    cdf_vals = ZIPoisson.cdf(x_vals, *p_params).eval()
    sf_vals = ZIPoisson.sf(x_vals, *p_params).eval()

    assert_allclose(cdf_vals + sf_vals, 1.0, rtol=1e-6, err_msg="CDF + SF != 1")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 2.0),
        (0.5, 5.0),
    ],
)
def test_zi_poisson_ppf_cdf_inverse(params):
    """Test that PPF is the inverse of CDF."""
    psi, mu = params
    p_params = make_params(psi, mu)

    # For discrete distributions, CDF(PPF(q)) >= q
    q_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    ppf_vals = ZIPoisson.ppf(q_vals, *p_params).eval()
    cdf_at_ppf = ZIPoisson.cdf(ppf_vals, *p_params).eval()

    assert np.all(cdf_at_ppf >= q_vals - 1e-10), "PPF-CDF inverse property violated"


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 2.0),
        (0.5, 5.0),
    ],
)
def test_zi_poisson_mean_variance(params):
    """Test theoretical mean and variance formulas."""
    psi, mu = params
    p_params = make_params(psi, mu)

    # Mean = psi * mu
    expected_mean = psi * mu
    actual_mean = ZIPoisson.mean(*p_params).eval()
    assert_allclose(actual_mean, expected_mean, rtol=1e-6, err_msg="Mean formula is incorrect")

    # Var = psi * (mu + (1-psi) * mu^2)
    expected_var = psi * (mu + (1 - psi) * mu**2)
    actual_var = ZIPoisson.var(*p_params).eval()
    assert_allclose(actual_var, expected_var, rtol=1e-6, err_msg="Variance formula is incorrect")


def test_zi_poisson_reduces_to_poisson():
    """Test that when psi=1, ZI-Poisson reduces to standard Poisson."""
    from pytensor_distributions import poisson as Poisson

    psi = 1.0
    mu = 3.0
    p_params = make_params(psi, mu)
    poisson_params = (pt.constant(mu, dtype="float64"),)

    x_vals = np.array([0, 1, 2, 3, 5, 10])

    # PMF should match
    zi_pmf = ZIPoisson.pdf(x_vals, *p_params).eval()
    poisson_pmf = Poisson.pdf(x_vals, *poisson_params).eval()
    assert_allclose(zi_pmf, poisson_pmf, rtol=1e-6, err_msg="ZI-Poisson(psi=1) != Poisson")

    # CDF should match
    zi_cdf = ZIPoisson.cdf(x_vals, *p_params).eval()
    poisson_cdf = Poisson.cdf(x_vals, *poisson_params).eval()
    assert_allclose(zi_cdf, poisson_cdf, rtol=1e-6, err_msg="ZI-Poisson CDF(psi=1) != Poisson CDF")


@pytest.mark.parametrize(
    "params",
    [
        (0.7, 2.0),
        (0.5, 5.0),
    ],
)
def test_zi_poisson_lmoments(params):
    psi, mu = params
    p_params = make_params(psi, mu)
    run_lmoments_test(p_dist=ZIPoisson, p_params=p_params, name="zi_poisson")
