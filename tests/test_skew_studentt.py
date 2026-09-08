"""Tests for Jones-Faddy Skew Student-t distribution against scipy.stats.jf_skew_t."""

import numpy as np
import pytest
from scipy.stats import jf_skew_t

from pytensor_distributions import skew_studentt as SkewStudentT
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params",
    [
        (5.0, 5.0, 0.0, 1.0),
        (
            np.array([8.0, 4.0, 10.0, 6.0]),
            np.array([4.0, 8.0, 5.0, 3.0]),
            np.array([0.0, 0.0, 2.0, -1.0]),
            np.array([1.0, 1.0, 3.0, 2.0]),
        ),
    ],
)
def test_skew_studentt_vs_scipy(params):
    """Test skew student-t distribution against scipy.stats.jf_skew_t."""
    a, b, mu, sigma = params
    p_params = make_params(a, b, mu, sigma, dtype="float64")
    sp_params = {
        "a": np.asarray(a),
        "b": np.asarray(b),
        "loc": np.asarray(mu),
        "scale": np.asarray(sigma),
    }

    run_distribution_tests(
        p_dist=SkewStudentT,
        sp_dist=jf_skew_t,
        p_params=p_params,
        sp_params=sp_params,
        support=[-np.inf, np.inf],
        name="skew_studentt",
        skip_mode=True,  # scipy doesn't provide mode for jf_skew_t
        use_quantiles_for_rvs=True,  # Compare quantiles instead of exact values
        entropy_rtol=1e-4,  # Numerical integration has ~0.002% error
        pdf_rtol=1e-5,
        logpdf_rtol=1e-5,
        cdf_rtol=1e-5,
        mean_rtol=1e-5,
        var_rtol=1e-5,
        std_rtol=1e-5,
        skewness_rtol=1e-5,
        kurtosis_rtol=1e-5,
    )


def test_skew_studentt_symmetric():
    """Test that a=b gives symmetric distribution (reduces to t-distribution behavior)."""
    a, b, mu, sigma = 5.0, 5.0, 0.0, 1.0
    p_params = make_params(a, b, mu, sigma, dtype="float64")

    # Mean should be mu (since a = b, no skewness)
    mean_val = SkewStudentT.mean(*p_params).eval()
    np.testing.assert_allclose(mean_val, 0.0, atol=1e-10)

    # Skewness should be 0
    skew_val = SkewStudentT.skewness(*p_params).eval()
    np.testing.assert_allclose(skew_val, 0.0, atol=1e-10)

    # PDF should be symmetric around mu
    pdf_left = SkewStudentT.pdf(-1.0, *p_params).eval()
    pdf_right = SkewStudentT.pdf(1.0, *p_params).eval()
    np.testing.assert_allclose(pdf_left, pdf_right, rtol=1e-10)


def test_skew_studentt_skew_direction():
    """Test that skewness direction matches a vs b relationship."""
    mu, sigma = 0.0, 1.0

    # Positive skew (a > b)
    a_pos, b_pos = 8.0, 4.0
    p_params_pos = make_params(a_pos, b_pos, mu, sigma, dtype="float64")
    skew_pos = SkewStudentT.skewness(*p_params_pos).eval()
    assert skew_pos > 0, f"Expected positive skewness when a > b, got {skew_pos}"

    # Negative skew (a < b)
    a_neg, b_neg = 4.0, 8.0
    p_params_neg = make_params(a_neg, b_neg, mu, sigma, dtype="float64")
    skew_neg = SkewStudentT.skewness(*p_params_neg).eval()
    assert skew_neg < 0, f"Expected negative skewness when a < b, got {skew_neg}"


def test_skew_studentt_batched():
    """Test that distribution handles batched parameters correctly."""
    a = np.array([5.0, 8.0, 4.0])
    b = np.array([5.0, 4.0, 8.0])
    mu = np.array([0.0, 1.0, -1.0])
    sigma = np.array([1.0, 2.0, 0.5])

    p_params = make_params(a, b, mu, sigma, dtype="float64")

    # Test that output shapes match input shapes
    mean_vals = SkewStudentT.mean(*p_params).eval()
    assert mean_vals.shape == (3,)

    # Test pdf with batched x
    x = np.array([0.0, 0.5, -0.5])
    pdf_vals = SkewStudentT.pdf(x, *p_params).eval()
    assert pdf_vals.shape == (3,)

    # Verify against individual scipy calls
    for i in range(3):
        expected = jf_skew_t.pdf(x[i], a[i], b[i], loc=mu[i], scale=sigma[i])
        np.testing.assert_allclose(pdf_vals[i], expected, rtol=1e-5)


def test_skew_studentt_large_ab():
    """Test behavior with large a and b (approaches normal distribution)."""
    a, b, mu, sigma = 100.0, 100.0, 0.0, 1.0
    p_params = make_params(a, b, mu, sigma, dtype="float64")

    # With large equal a and b, should approach normal distribution
    # Mean should be very close to mu
    mean_val = SkewStudentT.mean(*p_params).eval()
    np.testing.assert_allclose(mean_val, mu, atol=1e-2)

    # Skewness should be very close to 0
    skew_val = SkewStudentT.skewness(*p_params).eval()
    np.testing.assert_allclose(skew_val, 0.0, atol=1e-2)


def test_skew_studentt_moment_existence():
    """Test that moments return NaN when they don't exist."""
    mu, sigma = 0.0, 1.0

    # Mean doesn't exist when a <= 0.5 or b <= 0.5
    p_params = make_params(0.3, 5.0, mu, sigma, dtype="float64")
    mean_val = SkewStudentT.mean(*p_params).eval()
    assert np.isnan(mean_val)

    # Variance doesn't exist when a <= 1 or b <= 1
    p_params = make_params(0.8, 5.0, mu, sigma, dtype="float64")
    var_val = SkewStudentT.var(*p_params).eval()
    assert np.isnan(var_val)

    # Skewness doesn't exist when a <= 1.5 or b <= 1.5
    p_params = make_params(1.2, 5.0, mu, sigma, dtype="float64")
    skew_val = SkewStudentT.skewness(*p_params).eval()
    assert np.isnan(skew_val)

    # Kurtosis doesn't exist when a <= 2 or b <= 2
    p_params = make_params(1.8, 5.0, mu, sigma, dtype="float64")
    kurt_val = SkewStudentT.kurtosis(*p_params).eval()
    assert np.isnan(kurt_val)


def test_skew_studentt_mode():
    """Test that mode is at the peak of the PDF."""
    a, b, mu, sigma = 8.0, 4.0, 0.0, 1.0
    p_params = make_params(a, b, mu, sigma, dtype="float64")

    mode_val = SkewStudentT.mode(*p_params).eval()

    # PDF at mode should be >= PDF at nearby points
    eps = 0.01
    pdf_mode = SkewStudentT.pdf(mode_val, *p_params).eval()
    pdf_left = SkewStudentT.pdf(mode_val - eps, *p_params).eval()
    pdf_right = SkewStudentT.pdf(mode_val + eps, *p_params).eval()

    assert pdf_mode >= pdf_left - 1e-6, f"PDF at mode ({pdf_mode}) < PDF left ({pdf_left})"
    assert pdf_mode >= pdf_right - 1e-6, f"PDF at mode ({pdf_mode}) < PDF right ({pdf_right})"
