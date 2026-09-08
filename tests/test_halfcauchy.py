"""Test HalfCauchy distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import halfcauchy as HalfCauchy
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([3.5], {"scale": 3.5}),
        ([np.array([1e6, 1e-6])], {"scale": np.array([1e6, 1e-6])}),
    ],
)
def test_halfcauchy_vs_scipy(params, sp_params):
    """Test HalfCauchy distribution against scipy.stats.halfcauchy."""
    p_params = make_params(*params)
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=HalfCauchy,
        sp_dist=stats.halfcauchy,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="halfcauchy",
    )
