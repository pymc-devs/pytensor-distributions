"""Test ScaledInverseChiSquared distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import scaledinversechisquared as ScaledInverseChiSquared
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([4.0, 1.0], {"a": 2.0, "scale": 2.0}),
        ([10.0, 2.0], {"a": 5.0, "scale": 10.0}),
        ([3.5, 0.5], {"a": 1.75, "scale": 0.875}),
        ([8.0, 3.0], {"a": 4.0, "scale": 12.0}),
    ],
)
def test_scaledinversechisquared_vs_scipy(params, sp_params):
    """Test ScaledInverseChiSquared distribution against scipy inverse gamma."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=ScaledInverseChiSquared,
        sp_dist=stats.invgamma,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="scaledinversechisquared",
        use_quantiles_for_rvs=True,
    )
