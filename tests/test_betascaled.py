"""Test BetaScaled distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import betascaled as BetaScaled
from tests.helper_scipy import make_params, run_distribution_tests, run_lmoments_test


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 5.0, -1.0, 3.0], {"a": 2, "b": 5, "loc": -1, "scale": 4}),
        (
            [
                np.array([15.0, 20.0, 2.0]),
                np.array([3.0, 20.0, 5.0]),
                np.array([0.0, -100.0, -1.0]),
                np.array([10.0, 50.0, 3.0]),
            ],
            {
                "a": np.array([15.0, 20.0, 2.0]),
                "b": np.array([3.0, 20.0, 5.0]),
                "loc": np.array([0.0, -100.0, -1.0]),
                "scale": np.array([10.0, 150.0, 4.0]),
            },
        ),
    ],
)
def test_betascaled_vs_scipy(params, sp_params):
    """Test BetaScaled distribution against scipy.stats.beta."""
    p_params = make_params(*params, dtype="float64")
    lower, upper = np.min(params[2]), np.max(params[3])
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
