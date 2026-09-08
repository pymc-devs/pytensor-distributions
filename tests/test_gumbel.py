"""Test Gumbel distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import gumbel as Gumbel
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.5, 3.5], {"loc": 2.5, "scale": 3.5}),
        (
            [
                np.array([0.0, -1.0, -2.0, 100.0]),
                np.array([1.0, 2.0, 0.01, 100.0]),
            ],
            {
                "loc": np.array([0.0, -1.0, -2.0, 100.0]),
                "scale": np.array([1.0, 2.0, 0.01, 100.0]),
            },
        ),
    ],
)
def test_gumbel_vs_scipy(params, sp_params):
    """Test Gumbel distribution against scipy.stats.gumbel_r."""
    p_params = make_params(*params)
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=Gumbel,
        sp_dist=stats.gumbel_r,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="gumbel",
    )
