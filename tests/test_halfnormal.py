"""Test HalfNormal distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import halfnormal as HalfNormal
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([3.5], {"loc": 0, "scale": 3.5}),
        ([np.array([1e-6, 1e6])], {"loc": 0, "scale": np.array([1e-6, 1e6])}),
    ],
)
def test_halfnormal_vs_scipy(params, sp_params):
    """Test HalfNormal distribution against scipy.stats.halfnorm."""
    p_params = make_params(*params)
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=HalfNormal,
        sp_dist=stats.halfnorm,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="halfnormal",
    )
