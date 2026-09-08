"""Test Laplace distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import laplace as Laplace
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 1.0], {"loc": 0.0, "scale": 1.0}),
        ([-5.0, 0.5], {"loc": -5.0, "scale": 0.5}),
        ([10.0, 2.0], {"loc": 10.0, "scale": 2.0}),
        ([1.0, 1e-6], {"loc": 1.0, "scale": 1e-6}),
        (
            [
                np.array([0.0, -5.0]),
                np.array([1.0, 0.5]),
            ],
            {
                "loc": np.array([0.0, -5.0]),
                "scale": np.array([1.0, 0.5]),
            },
        ),
    ],
)
def test_laplace_vs_scipy(params, sp_params):
    """Test Laplace distribution against scipy."""
    p_params = make_params(*params)
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=Laplace,
        sp_dist=stats.laplace,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="laplace",
    )
