"""Test DiscreteUniform distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import discreteuniform as DiscreteUniform
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params, use_quantiles",
    [
        ([-2, 1], {"low": -2, "high": 2}, False),
        ([0, 5], {"low": 0, "high": 6}, False),
        (
            [
                np.array([1, -10], dtype="int64"),
                np.array([10, -5], dtype="int64"),
            ],
            {
                "low": np.array([1, -10]),
                "high": np.array([11, -4]),
            },
            True,
        ),
    ],
)
def test_discreteuniform_vs_scipy(params, sp_params, use_quantiles):
    """Test DiscreteUniform distribution against scipy.stats.randint."""
    lower, upper = params
    p_params = make_params(lower, upper, dtype="int64")
    support = (int(np.min(lower)), int(np.max(upper)))

    run_distribution_tests(
        p_dist=DiscreteUniform,
        sp_dist=stats.randint,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="discreteuniform",
        is_discrete=True,
        skip_mode=True,
        use_quantiles_for_rvs=use_quantiles,
    )
