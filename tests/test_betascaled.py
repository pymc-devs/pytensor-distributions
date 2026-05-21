"""Test BetaScaled distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import betascaled as BetaScaled
from tests.helper_scipy import make_params, run_distribution_tests, run_lmoments_test


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 5.0, -1.0, 3.0], {"a": 2, "b": 5, "loc": -1, "scale": 4}),
        ([15.0, 3.0, 0.0, 10.0], {"a": 15.0, "b": 3.0, "loc": 0.0, "scale": 10.0}),
        ([20.0, 20.0, -100.0, 50.0], {"a": 20.0, "b": 20.0, "loc": -100.0, "scale": 150.0}),
    ],
)
def test_betascaled_vs_scipy(params, sp_params):
    """Test BetaScaled distribution against scipy.stats.beta."""
    p_params = make_params(*params, dtype="float64")
    lower, upper = params[2], params[3]
    support = (lower, upper)

    run_distribution_tests(
        p_dist=BetaScaled,
        sp_dist=stats.beta,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="betascaled",
    )
    run_lmoments_test(p_dist=BetaScaled, p_params=p_params, name="betascaled")
