"""Test Logit-Normal distribution against empirical samples."""

import pytest

from pytensor_distributions import logitnormal as LogitNormal
from tests.helper_empirical import run_empirical_tests
from tests.helper_scipy import make_params, run_lmoments_test


@pytest.mark.parametrize(
    "params",
    [
        [0.0, 1.0],  # Standard logit-normal (centered)
        [0.0, 0.001],  # Narrower distribution
        [-1.0, 1.0],  # Shifted left (mode < 0.5)
        [0.0, 2.0],  # Wider distribution (approaches U-shape)
        [2.0, 0.5],  # Strongly shifted right
    ],
)
def test_logitnormal_vs_random(params):
    """Test Logit-Normal distribution against random samples."""
    p_params = make_params(*params, dtype="float64")
    support = (0, 1)

    run_empirical_tests(
        p_dist=LogitNormal,
        p_params=p_params,
        support=support,
        name="logitnormal",
        mean_rtol=1e-2,
        var_rtol=1e-2,
        std_rtol=1e-2,
        skewness_rtol=2e-1,
        kurtosis_rtol=2e-1 if params[1] > 0.01 else 1,
        quantiles_rtol=3e-2,
        cdf_rtol=5e-2,
        pdf_cdf_rtol=1e-2,
    )
    run_lmoments_test(p_dist=LogitNormal, p_params=p_params, name="logitnormal")
