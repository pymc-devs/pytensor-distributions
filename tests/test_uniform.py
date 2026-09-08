"""Test Uniform distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import uniform as Uniform
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 1.0], {"loc": 0.0, "scale": 1.0}),
        (
            [
                np.array([-5.0, 10.0, -100.0, 0.0]),
                np.array([5.0, 20.0, -50.0, 1e-6]),
            ],
            {
                "loc": np.array([-5.0, 10.0, -100.0, 0.0]),
                "scale": np.array([10.0, 10.0, 50.0, 1e-6]),
            },
        ),
    ],
)
def test_uniform_vs_scipy(params, sp_params):
    """Test Uniform distribution against scipy."""
    p_params = make_params(*params)
    support = (float(np.min(params[0])), float(np.max(params[1])))

    run_distribution_tests(
        p_dist=Uniform,
        sp_dist=stats.uniform,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="uniform",
    )
