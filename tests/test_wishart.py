"""Test Wishart distribution."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import wishart as scipy_wishart

from pytensor_distributions import wishart as Wishart

# Test cases: (nu, V)
TEST_CASES = [
    (5, np.array([[2.0, 0.5], [0.5, 1.0]])),
    (10, np.array([[1.0, 0.0], [0.0, 1.0]])),
    (7, np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.5]])),
]


@pytest.mark.parametrize("nu, V", TEST_CASES)
def test_wishart_mean(nu, V):
    scipy_dist = scipy_wishart(df=nu, scale=V)

    p_nu = pt.constant(nu)
    p_V = pt.constant(V)

    actual = Wishart.mean(p_nu, p_V).eval()
    expected = scipy_dist.mean()
    assert_allclose(actual, expected, rtol=1e-10, err_msg=f"Mean should match scipy for nu={nu}")


@pytest.mark.parametrize("nu, V", TEST_CASES)
def test_wishart_mode(nu, V):
    p = V.shape[0]
    p_nu = pt.constant(nu)
    p_V = pt.constant(V)

    actual = Wishart.mode(p_nu, p_V).eval()
    expected = (nu - p - 1) * V
    assert_allclose(actual, expected, rtol=1e-10, err_msg=f"Mode should match formula for nu={nu}")


@pytest.mark.parametrize("nu, V", TEST_CASES)
def test_wishart_var(nu, V):
    scipy_dist = scipy_wishart(df=nu, scale=V)
    p_nu = pt.constant(nu)
    p_V = pt.constant(V)

    actual = Wishart.var(p_nu, p_V).eval()
    expected = np.diag(scipy_dist.var())
    assert_allclose(
        actual, expected, rtol=1e-10, err_msg=f"Variance should match scipy for nu={nu}"
    )


@pytest.mark.parametrize("nu, V", TEST_CASES)
def test_wishart_entropy(nu, V):
    scipy_dist = scipy_wishart(df=nu, scale=V)

    p_nu = pt.constant(nu)
    p_V = pt.constant(V)

    actual = Wishart.entropy(p_nu, p_V).eval()
    expected = scipy_dist.entropy()
    assert_allclose(actual, expected, rtol=1e-5, err_msg=f"Entropy should match scipy for nu={nu}")


@pytest.mark.parametrize("nu, V", TEST_CASES)
def test_wishart_logpdf(nu, V):
    scipy_dist = scipy_wishart(df=nu, scale=V)
    p_nu = pt.constant(nu)
    p_V = pt.constant(V)

    X = scipy_dist.mean()

    actual = Wishart.logpdf(X, p_nu, p_V).eval()
    expected = scipy_dist.logpdf(X)
    assert_allclose(
        actual, expected, rtol=1e-5, err_msg=f"logpdf at mean should match scipy for nu={nu}"
    )


@pytest.mark.parametrize("nu, V", TEST_CASES)
def test_wishart_pdf(nu, V):
    scipy_dist = scipy_wishart(df=nu, scale=V)

    p_nu = pt.constant(nu)
    p_V = pt.constant(V)

    X = scipy_dist.mean()

    actual = Wishart.pdf(X, p_nu, p_V).eval()
    expected = scipy_dist.pdf(X)
    assert_allclose(actual, expected, rtol=1e-5, err_msg=f"pdf should match scipy for nu={nu}")


def test_wishart_constraints():
    """Test that logpdf returns -inf for invalid parameters."""
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = np.array([[1.0, 0.0], [0.0, 1.0]])

    p_V = pt.constant(V)
    p_X = pt.constant(X)

    invalid_nu = pt.constant(1.0)
    actual = Wishart.logpdf(p_X, invalid_nu, p_V).eval()
    assert actual == -np.inf, "logpdf should be -inf when nu <= p - 1"


@pytest.mark.parametrize("nu, V", TEST_CASES)
def test_wishart_rvs(nu, V):
    """Test Wishart random sampling: shape, moments, and positive definiteness."""
    p_nu = pt.constant(nu)
    p_V = pt.constant(V)
    p = V.shape[0]

    sample = Wishart.rvs(p_nu, p_V, size=None).eval()
    assert sample.shape == (p, p), f"Single sample should have shape ({p}, {p})"
    eigenvalues = np.linalg.eigvalsh(sample)
    assert np.all(eigenvalues > 0), "Single sample should be positive definite"

    n_samples = 1000
    samples = Wishart.rvs(p_nu, p_V, size=n_samples).eval()
    assert samples.shape == (
        n_samples,
        p,
        p,
    ), f"Multiple samples should have shape ({n_samples}, {p}, {p})"

    for i in range(min(10, n_samples)):
        eigenvalues = np.linalg.eigvalsh(samples[i])
        assert np.all(eigenvalues > 0), f"Sample {i} should be positive definite"

    sample_mean = np.mean(samples, axis=0)
    theoretical_mean = Wishart.mean(p_nu, p_V).eval()
    assert_allclose(
        sample_mean,
        theoretical_mean,
        rtol=0.15,
        atol=0.2,
        err_msg=f"Sample mean should match theoretical mean for nu={nu}",
    )

    sample_var = np.var(samples, axis=0)
    sample_var_diag = np.diag(sample_var)
    theoretical_var = Wishart.var(p_nu, p_V).eval()
    assert_allclose(
        sample_var_diag,
        theoretical_var,
        rtol=0.2,
        atol=0.1,
        err_msg=f"Sample variance should match theoretical variance for nu={nu}",
    )
