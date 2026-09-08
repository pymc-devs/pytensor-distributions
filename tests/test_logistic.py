"""Test Logistic distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import logistic as Logistic
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.5, 4], {"loc": 2.5, "scale": 4}),
        (
            [
                np.array([-1.0, 0.0, 100.0]),
                np.array([2.0, 0.01, 10.0]),
            ],
            {
                "loc": np.array([-1.0, 0.0, 100.0]),
                "scale": np.array([2.0, 0.01, 10.0]),
            },
        ),
    ],
)
def test_logistic_vs_scipy(params, sp_params):
    """Test Logistic distribution against scipy.stats.logistic."""
    p_params = make_params(*params)
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=Logistic,
        sp_dist=stats.logistic,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="logistic",
    )
