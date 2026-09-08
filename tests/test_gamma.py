"""Test Gamma distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import gamma as Gamma
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 1 / 3], {"a": 2.0, "scale": 3.0}),
        (
            [
                np.array([1.0, 0.5, 50.0, 0.01, 100.0]),
                np.array([1.0, 2.0, 0.5, 0.01, 100.0]),
            ],
            {
                "a": np.array([1.0, 0.5, 50.0, 0.01, 100.0]),
                "scale": np.array([1.0, 0.5, 2.0, 100.0, 0.01]),
            },
        ),
    ],
)
def test_gamma_vs_scipy(params, sp_params):
    """Test Gamma distribution against scipy.stats.gamma."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=Gamma,
        sp_dist=stats.gamma,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="gamma",
    )
