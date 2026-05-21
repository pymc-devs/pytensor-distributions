"""Tests for the HalfStudentT distribution."""

import pytest
from scipy import stats

from pytensor_distributions import halfstudentt as HalfStudentT
from tests.helper_scipy import make_params, run_distribution_tests

# HalfStudentT is not defined in scipy
test_cases = [
    # Very high nu, some functions use a HalfNormal approximation
    {
        "params": [1e6, 2.0],
        "sp_dist": stats.halfnorm,
        "sp_params": {"loc": 0, "scale": 2},
        "name": "halfstudent_high_nu_approx",
        "special_settings": {
            "skip_skewness": True,
            "skip_kurtosis": True,
            "use_quantiles_for_rvs": True,
        },
    },
    # Large nu but below halfnormal approximation
    {
        "params": [1e4, 2.0],
        "sp_dist": stats.halfnorm,
        "sp_params": {"loc": 0, "scale": 2},
        "name": "halfstudent_large_nu_exact",
        "special_settings": {
            "entropy_rtol": 1e-2,
            "pdf_rtol": 1e-2,
            "logpdf_rtol": 1e-2,
            "cdf_rtol": 1e-2,
            "logcdf_rtol": 1e-2,
            "sf_rtol": 1e-2,
            "logsf_rtol": 1e-2,
            "isf_rtol": 1e-2,
            "median_rtol": 1e-2,
            "skip_skewness": True,
            "skip_kurtosis": True,
            "use_quantiles_for_rvs": True,
        },
    },
    # nu=1, HalfCauchy
    {
        "params": [1.0, 3.5],
        "sp_dist": stats.halfcauchy,
        "sp_params": {"scale": 3.5},
        "name": "halfstudentt",
        "special_settings": {
            "use_quantiles_for_rvs": True,
        },
    },
]


@pytest.mark.parametrize("test_case", test_cases)
def test_halfstudentt_vs_scipy(test_case):
    """Test HalfStudentT distribution against appropriate scipy distributions."""
    special_settings = test_case.get("special_settings", {})
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=HalfStudentT,
        sp_dist=test_case["sp_dist"],
        p_params=make_params(*test_case["params"], dtype="float64"),
        sp_params=test_case["sp_params"],
        support=support,
        name=test_case["name"],
        **special_settings,
    )
