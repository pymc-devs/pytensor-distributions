"""Test DiscreteUniform distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import discreteuniform as DiscreteUniform
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([-2, 1], {"low": -2, "high": 2}),
        ([0, 5], {"low": 0, "high": 6}),
        ([-10, -5], {"low": -10, "high": -4}),
        ([1, 10], {"low": 1, "high": 11}),
    ],
)
def test_discreteuniform_vs_scipy(params, sp_params):
    """Test DiscreteUniform distribution against scipy.stats.randint."""
    lower, upper = params
    p_params = make_params(lower, upper, dtype="int64")
    support = (lower, upper)

    run_distribution_tests(
        p_dist=DiscreteUniform,
        sp_dist=stats.randint,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="discreteuniform",
        is_discrete=True,
        skip_mode=True,  # Mode is undefined (all values equally likely)
    )
