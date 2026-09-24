"""Tail-precision regression tests for isf/ppf/sf/logsf.

These tests verify that the fixed implementations maintain accuracy in the
extreme tails where naive formulations break down (e.g., erfinv(1-2q) at q=1e-20).
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy import special

from pytensor_distributions import (
    cauchy,
    exgaussian,
    geometric,
    halfstudentt,
    laplace,
    lognormal,
    moyal,
    normal,
    pareto,
    rice,
    skew_studentt,
    skewnormal,
    studentt,
    truncatednormal,
    uniform,
    wald,
)

EXTREME_Q = np.array([1e-20, 1e-16, 1e-8, 1e-4])


def assert_roundtrip(isf_fn, sf_fn, args, qs):
    """sf(isf(q)) must recover q for the quantiles the root finder claims."""
    x = isf_fn(qs, *args).eval()
    s = sf_fn(x, *args).eval()
    assert_allclose(s, qs, rtol=1e-6, atol=0)


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


class TestClosedFormIsfTailPrecision:
    """isf implemented with a closed form must be finite and invert sf.

    Each of these previously delegated to ``ppf(1 - q)``, which saturates to
    the support bound (or overflows) for q <= 1e-16.
    """

    CASES = [
        ("moyal", moyal.isf, moyal.sf, (0.0, 1.0)),
        ("laplace", laplace.isf, laplace.sf, (0.0, 1.0)),
        ("pareto", pareto.isf, pareto.sf, (2.5, 1.0)),
        ("studentt", studentt.isf, studentt.sf, (4.0, 0.0, 1.0)),
        ("halfstudentt", halfstudentt.isf, halfstudentt.sf, (4.0, 1.0)),
        ("uniform", uniform.isf, uniform.sf, (0.0, 1.0)),
        ("wald", wald.isf, wald.sf, (2.0, 3.0)),
    ]

    def test_isf_finite_at_extreme_q(self):
        for name, isf_fn, _sf, args in self.CASES:
            result = isf_fn(EXTREME_Q, *args).eval()
            assert np.all(np.isfinite(result)), f"{name}.isf not finite: {result}"
            assert np.all(result != -np.inf), f"{name}.isf returned -inf"

    def test_isf_sf_roundtrip(self):
        # wald/skew-rice/exgaussian use a root finder on sf = 1 - cdf, which
        # floors near q ~ 1e-16, so only the closed-form cases get the full
        # extreme-quantile roundtrip.
        for name, isf_fn, sf_fn, args in [
            ("moyal", moyal.isf, moyal.sf, (0.0, 1.0)),
            ("laplace", laplace.isf, laplace.sf, (0.0, 1.0)),
            ("pareto", pareto.isf, pareto.sf, (2.5, 1.0)),
            ("studentt", studentt.isf, studentt.sf, (4.0, 0.0, 1.0)),
            ("halfstudentt", halfstudentt.isf, halfstudentt.sf, (4.0, 1.0)),
            ("wald", wald.isf, wald.sf, (2.0, 3.0)),
        ]:
            try:
                assert_roundtrip(isf_fn, sf_fn, args, EXTREME_Q)
            except AssertionError as exc:
                raise AssertionError(f"{name}: {exc}") from None

    def test_isf_monotone_decreasing_in_q(self):
        q = np.array([1e-20, 1e-16, 1e-8, 1e-4, 0.5, 0.99])
        for name, isf_fn, _sf, args in self.CASES:
            result = isf_fn(q, *args).eval()
            assert np.all(np.diff(result) < 0), f"{name}.isf not decreasing: {result}"

    def test_sf_decays_at_large_x(self):
        x = np.array([1e3, 1e6])
        for name, _isf, sf_fn, args in self.CASES:
            s = sf_fn(x, *args).eval()
            assert np.all(s >= 0), f"{name}.sf negative: {s}"
            # uniform.sf is exactly 0 beyond the bound, so allow a plateau
            assert np.all(np.diff(s) <= 0), f"{name}.sf increasing: {s}"

    def test_sf_closed_forms(self):
        x = np.array([1e3, 1e6])
        # pareto: sf = (m / x) ** alpha
        s = pareto.sf(x, 2.5, 1.0).eval()
        assert_allclose(s, (1.0 / x) ** 2.5, rtol=1e-12)
        # laplace: sf = 0.5 * exp(-(x - mu) / b)
        s = laplace.sf(x, 0.0, 1.0).eval()
        assert_allclose(s, 0.5 * np.exp(-x), rtol=1e-12)

    def test_logsf_finite_at_large_x(self):
        x = np.array([1e3, 1e6])
        for name, module, args in [
            ("laplace", laplace, (0.0, 1.0)),
            ("pareto", pareto, (2.5, 1.0)),
            ("studentt", studentt, (4.0, 0.0, 1.0)),
            ("halfstudentt", halfstudentt, (4.0, 1.0)),
        ]:
            result = module.logsf(x, *args).eval()
            assert np.all(np.isfinite(result)), f"{name}.logsf not finite: {result}"
            assert np.all(result < 0), f"{name}.logsf not negative: {result}"


class TestRootFindIsfTailPrecision:
    """isf solved by root-finding must still be finite at extreme q.

    exgaussian/rice/skewnormal cannot use a closed form, so accuracy floors
    near q ~ 1e-16 (sf = 1 - cdf cancellation); the requirement here is that
    the old ``ppf(1 - q)`` failure mode (inf) is gone and that moderate
    quantiles still round-trip.
    """

    CASES = [
        ("exgaussian", exgaussian.isf, exgaussian.sf, (0.0, 1.0, 2.0)),
        ("rice", rice.isf, rice.sf, (2.0, 1.0)),
        ("skewnormal", skewnormal.isf, skewnormal.sf, (0.0, 1.0, 3.0)),
    ]

    def test_isf_finite_at_extreme_q(self):
        for name, isf_fn, _sf, args in self.CASES:
            result = isf_fn(EXTREME_Q, *args).eval()
            assert np.all(np.isfinite(result)), f"{name}.isf not finite: {result}"

    def test_isf_sf_roundtrip_moderate_q(self):
        qs = np.array([1e-8, 1e-4, 0.01, 0.5, 0.99])
        for name, isf_fn, sf_fn, args in self.CASES:
            try:
                assert_roundtrip(isf_fn, sf_fn, args, qs)
            except AssertionError as exc:
                raise AssertionError(f"{name}: {exc}") from None

    def test_isf_monotone_decreasing_in_q(self):
        q = np.array([1e-8, 1e-4, 0.01, 0.5, 0.99])
        for name, isf_fn, _sf, args in self.CASES:
            result = isf_fn(q, *args).eval()
            assert np.all(np.diff(result) <= 0), f"{name}.isf not decreasing: {result}"


class TestTruncatedNormalTailPrecision:
    """truncatednormal.isf must saturate at the finite upper bound, not inf."""

    ARGS = (0.0, 1.0, -2.0, 3.0)

    def test_isf_finite_at_extreme_q(self):
        result = truncatednormal.isf(EXTREME_Q, *self.ARGS).eval()
        assert np.all(np.isfinite(result))
        # for tiny q the quantile is within float resolution of the bound
        assert np.all(result <= 3.0 + 1e-12)

    def test_isf_sf_roundtrip_moderate_q(self):
        qs = np.array([1e-8, 1e-4, 0.01, 0.5, 0.99])
        assert_roundtrip(truncatednormal.isf, truncatednormal.sf, self.ARGS, qs)

    def test_sf_outside_support(self):
        s = truncatednormal.sf(np.array([-3.0, -2.5]), *self.ARGS).eval()
        assert_allclose(s, [1.0, 1.0])
        s_hi = truncatednormal.sf(np.array([3.5, 1e3]), *self.ARGS).eval()
        assert_allclose(s_hi, [0.0, 0.0])


class TestDiscreteIsfTailPrecision:
    """Discrete isf must satisfy the survival-window property sf(x) <= q < sf(x-1)."""

    CASES = [
        ("geometric", geometric.isf, geometric.sf, (0.3,)),
    ]

    def test_isf_window(self):
        for name, isf_fn, sf_fn, args in self.CASES:
            for q in (1e-16, 1e-6, 0.01, 0.5):
                x = isf_fn(np.array([q]), *args).eval()
                assert np.all(np.isfinite(x))
                s = sf_fn(x, *args).eval()
                s_prev = sf_fn(x - 1, *args).eval()
                assert np.all(s <= q * (1 + 1e-9) + 1e-300), f"{name} q={q}: sf(isf(q))={s} > q"
                assert np.all(s_prev > q * (1 - 1e-9) - 1e-300), (
                    f"{name} q={q}: sf(isf(q)-1)={s_prev} <= q"
                )

    def test_isf_endpoints(self):
        # q=0 -> +inf (last support point), q=1 -> lower - 1 = 0 for geometric
        lo = geometric.isf(np.array([1.0]), 0.3).eval()
        assert np.all(lo == 0.0)
        hi = geometric.isf(np.array([0.0]), 0.3).eval()
        assert np.all(np.isinf(hi))
