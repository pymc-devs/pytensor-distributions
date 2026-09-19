"""Tail-precision regression tests for isf/ppf/sf/logsf.

These tests verify that the fixed implementations maintain accuracy in the
extreme tails where naive formulations break down (e.g., erfinv(1-2q) at q=1e-20).
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy import special

from pytensor_distributions import cauchy, lognormal, normal, skew_studentt


class TestNormalTailPrecision:
    """normal.ppf/isf must not return inf/-inf at extreme quantiles."""

    def test_ppf_small_q(self):
        q = np.array([1e-10, 1e-16, 1e-20])
        result = normal.ppf(q, 0.0, 1.0).eval()
        expected = special.ndtri(q)
        assert_allclose(result, expected, rtol=1e-10)

    def test_isf_small_q(self):
        q = np.array([1e-10, 1e-16, 1e-20])
        result = normal.isf(q, 0.0, 1.0).eval()
        expected = -special.ndtri(q)
        assert_allclose(result, expected, rtol=1e-10)

    def test_sf_large_z(self):
        z = np.array([6.0, 10.0, 30.0])
        result = normal.sf(z, 0.0, 1.0).eval()
        expected = 0.5 * special.erfc(z / 2**0.5)
        assert_allclose(result, expected, rtol=1e-10)


class TestLognormalTailPrecision:
    """lognormal.ppf/isf must not return 0/inf at extreme quantiles."""

    def test_ppf_small_q(self):
        q = np.array([1e-10, 1e-16, 1e-20])
        result = lognormal.ppf(q, 0.0, 1.0).eval()
        expected = np.exp(special.ndtri(q))
        assert_allclose(result, expected, rtol=1e-10)

    def test_isf_small_q(self):
        q = np.array([1e-10, 1e-16, 1e-20])
        result = lognormal.isf(q, 0.0, 1.0).eval()
        expected = np.exp(-special.ndtri(q))
        assert_allclose(result, expected, rtol=1e-10)


class TestCauchyTailPrecision:
    """cauchy.ppf/isf must not suffer from tan(pi*(q-0.5)) pole amplification."""

    def test_ppf_small_q(self):
        q = np.array([1e-10, 1e-16, 1e-20])
        result = cauchy.ppf(q, 0.0, 1.0).eval()
        expected = -1.0 / np.tan(np.pi * q)
        assert_allclose(result, expected, rtol=1e-10)

    def test_isf_small_q(self):
        q = np.array([1e-10, 1e-16, 1e-20])
        result = cauchy.isf(q, 0.0, 1.0).eval()
        expected = 1.0 / np.tan(np.pi * q)
        assert_allclose(result, expected, rtol=1e-10)


class TestSkewStudenttTailPrecision:
    """skew_studentt.sf/logsf must not return 0/-inf at large z."""

    def test_sf_large_z(self):
        z = np.array([500.0, 1e3, 1e6])
        x = z * 1.0 + 0.0
        result = skew_studentt.sf(x, 2.0, 3.0, 0.0, 1.0).eval()
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_logsf_large_z(self):
        z = np.array([500.0, 1e3, 1e6])
        x = z * 1.0 + 0.0
        result = skew_studentt.logsf(x, 2.0, 3.0, 0.0, 1.0).eval()
        assert np.all(np.isfinite(result))
        assert np.all(result < 0)
