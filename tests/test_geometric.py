"""Test Geometric distribution against scipy implementation."""

import numpy as np
import pytensor.tensor as pt
import pytest
from scipy import stats

from pytensor_distributions import geometric as Geometric
from tests.helper_scipy import run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.1], {"p": 0.1}),
        ([np.array([0.5, 0.9])], {"p": np.array([0.5, 0.9])}),
    ],
)
def test_geometric_vs_scipy(params, sp_params):
    """Test Geometric distribution against scipy.stats.geom."""
    p_param = pt.constant(params[0], dtype="float64")
    p_params = (p_param,)
    support = (1, float("inf"))

    run_distribution_tests(
        p_dist=Geometric,
        sp_dist=stats.geom,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="geometric",
        is_discrete=True,
    )
