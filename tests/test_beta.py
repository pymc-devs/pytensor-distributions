"""Test Beta distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import beta as Beta
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params, skip_mode",
    [
        ([2.0, 5.0], {"a": 2, "b": 5}, False),
        ([100.0, 100.0], {"a": 100.0, "b": 100.0}, False),
        (
            [
                np.array([0.5, 1.0]),
                np.array([3.0, 1.0]),
            ],
            {
                "a": np.array([0.5, 1.0]),
                "b": np.array([3.0, 1.0]),
            },
            True,
        ),
    ],
)
def test_beta_vs_scipy(params, sp_params, skip_mode):
    """Test Beta distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (0, 1)

    run_distribution_tests(
        p_dist=Beta,
        sp_dist=stats.beta,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="beta",
        skip_mode=skip_mode,
    )
