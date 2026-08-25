"""Test Beta Prime distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import betaprime as BetaPrime
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([3.0, 5.0], {"a": 3.0, "b": 5}),
        ([12, 2.0], {"a": 12, "b": 2}),
        ([75.0, 20.0], {"a": 75.0, "b": 20.0}),
        ([2.5, 4.5], {"a": 2.5, "b": 4.5}),
    ],
)
def test_betaprime_vs_scipy(params, sp_params):
    """Test Beta distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (0, 1)

    run_distribution_tests(
        p_dist=BetaPrime,
        sp_dist=stats.betaprime,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="betaprime",
        use_quantiles_for_rvs=True,
    )
