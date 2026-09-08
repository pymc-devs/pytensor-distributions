"""Test Moyal distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import moyal as Moyal
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([-1e6, 100.0], {"loc": -1e6, "scale": 100.0}),
        ([10.0, 1e-3], {"loc": 10.0, "scale": 1e-3}),
        ([1.0, 1e-4], {"loc": 1.0, "scale": 1e-4}),
        (
            [
                np.array([0.0, -1]),
                np.array([1.0, 0.5]),
            ],
            {
                "loc": np.array([0.0, -1]),
                "scale": np.array([1.0, 0.5]),
            },
        ),
    ],
)
def test_moyal_vs_scipy(params, sp_params):
    """Test Moyal distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=Moyal,
        sp_dist=stats.moyal,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="moyal",
        use_quantiles_for_rvs=True,
    )
