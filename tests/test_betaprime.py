"""Test Beta Prime distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import betaprime as BetaPrime
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params, skip_mode",
    [
        ([3.0, 5.0], {"a": 3.0, "b": 5}, False),
        ([12, 2.0], {"a": 12, "b": 2}, False),
        ([75.0, 20.0], {"a": 75.0, "b": 20.0}, False),
        ([2.75, 0.75], {"a": 2.75, "b": 0.75}, False),
        ([0.5, 0.5], {"a": 0.5, "b": 0.5}, True),
    ],
)
def test_betaprime_vs_scipy(params, sp_params, skip_mode):
    """Test BetaPrime distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=BetaPrime,
        sp_dist=stats.betaprime,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="betaprime",
        skip_mode=skip_mode,
    )
