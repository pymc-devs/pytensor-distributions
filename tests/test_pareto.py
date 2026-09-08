"""Test Pareto distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import pareto as Pareto
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([3.0, 1.0], {"b": 3.0, "scale": 1.0}),
        (
            [
                np.array([2.5, 4.1, 1.5, 0.5]),
                np.array([2.0, 0.5, 10.0, 1.0]),
            ],
            {
                "b": np.array([2.5, 4.1, 1.5, 0.5]),
                "scale": np.array([2.0, 0.5, 10.0, 1.0]),
            },
        ),
    ],
)
def test_pareto_vs_scipy(params, sp_params):
    """Test Pareto distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (float(np.min(params[1])), float("inf"))

    run_distribution_tests(
        p_dist=Pareto,
        sp_dist=stats.pareto,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="pareto",
    )
