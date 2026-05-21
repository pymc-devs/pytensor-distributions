"""Test LogNormal distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import lognormal as LogNormal
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 0.5], {"s": 0.5, "loc": 0, "scale": np.exp(2.0)}),
        ([0.0, 1.0], {"s": 1.0, "loc": 0, "scale": 1.0}),
        ([-1.0, 0.25], {"s": 0.25, "loc": 0, "scale": np.exp(-1.0)}),
        ([0.0, 0.001], {"s": 0.001, "loc": 0, "scale": 1.0}),
        ([5.0, 1.5], {"s": 1.5, "loc": 0, "scale": np.exp(5.0)}),
    ],
)
def test_lognormal_vs_scipy(params, sp_params):
    """Test LogNormal distribution against scipy.stats.lognorm."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=LogNormal,
        sp_dist=stats.lognorm,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="lognormal",
    )
