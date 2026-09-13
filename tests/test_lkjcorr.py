"""Test LKJ distribution."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pytensor_distributions import lkjcorr as LKJ

TEST_CASES = [
    (2, 1.0),
    (2, 3.5),
    (3, 2.0),
    (4, 1.5),
    (5, 3.0),
    (6, 0.8),
]


# Test for K=2
@pytest.mark.parametrize("K, eta", [(2, eta) for eta in (0.5, 1.0, 1.3, 2.0, 5.0)])
@pytest.mark.parametrize("r", [-0.8, -0.3, 0.0, 0.4, 0.9])
def test_lkj_pdf_k2(K, eta, r):
    C = np.array([[1.0, r], [r, 1.0]])
    p_C = pt.constant(C)
    p_eta = pt.constant(eta)

    actual = LKJ.pdf(p_C, K, p_eta).eval()
    expected = np.exp(stats.beta(eta, eta).logpdf((r + 1) / 2) + np.log(0.5))
    assert_allclose(actual, expected, rtol=1e-3)


@pytest.mark.parametrize("K, eta", TEST_CASES)
def test_lkj_mean(K, eta):
    actual = LKJ.mean(K, pt.constant(eta)).eval()
    assert_allclose(actual, np.eye(K))


@pytest.mark.parametrize("K, eta", TEST_CASES)
def test_lkj_var(K, eta):
    actual = LKJ.var(K, pt.constant(eta)).eval()
    expected = (4 * (eta + K / 2 - 1) ** 2) / ((2 * eta + K - 2) ** 2 * (2 * eta + K - 1))
    assert_allclose(actual, expected, rtol=1e-3)


def test_lkj_constraints():
    """Logpdf should be -inf (or nan-safe) for eta <= 0."""
    C = np.eye(3)
    p_C = pt.constant(C)
    invalid_eta = pt.constant(-1.0)

    actual = LKJ.logpdf(p_C, 3, invalid_eta).eval()
    assert not np.isfinite(actual), "logpdf should not be finite when eta <= 0"


@pytest.mark.parametrize("K, eta", TEST_CASES)
def test_lkj_rvs(K, eta):
    """Test LKJ random sampling."""
    p_eta = pt.constant(eta)

    sample = LKJ.rvs(K, p_eta, size=None).eval()
    assert_allclose(sample.shape, (K, K))
    assert_allclose(np.diagonal(sample), 1.0, rtol=1e-8)
    assert_allclose(sample, sample.T, rtol=1e-8)

    eigenvalues = np.linalg.eigvalsh(sample)
    assert_allclose(np.minimum(eigenvalues, 0.0), 0.0, rtol=1e-8)

    n_samples = 200_000
    samples = LKJ.rvs(K, p_eta, size=n_samples).eval()
    assert_allclose(samples.shape, (n_samples, K, K))

    for i in range(min(10, n_samples)):
        eigenvalues = np.linalg.eigvalsh(samples[i])
        assert_allclose(np.minimum(eigenvalues, 0.0), 0.0, rtol=1e-8)

    sample_mean = samples.mean(axis=0)
    assert_allclose(sample_mean, np.eye(K), atol=1e-2)

    theoretical_var = LKJ.var(K, p_eta).eval()
    iu = np.triu_indices(K, k=1)
    offdiag_samples = samples[:, iu[0], iu[1]]
    sample_var = offdiag_samples.var()
    assert_allclose(sample_var, theoretical_var, rtol=1e-2)
