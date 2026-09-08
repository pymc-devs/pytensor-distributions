"""Test BetaBinomial distribution against scipy implementation."""

import numpy as np
import pytensor.tensor as pt
import pytest
from scipy import stats

from pytensor_distributions import betabinomial as BetaBinomial
from tests.helper_scipy import run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params, skip_mode",
    [
        ([10, 2.0, 3.0], {"n": 10, "a": 2.0, "b": 3.0}, False),
        (
            [
                np.array([6, 20, 15, 100, 10, 10], dtype="int64"),
                np.array([1.0, 0.5, 5.0, 20.0, 1.0, 3.0]),
                np.array([1.0, 0.5, 2.0, 20.0, 3.0, 1.0]),
            ],
            {
                "n": np.array([6, 20, 15, 100, 10, 10]),
                "a": np.array([1.0, 0.5, 5.0, 20.0, 1.0, 3.0]),
                "b": np.array([1.0, 0.5, 2.0, 20.0, 3.0, 1.0]),
            },
            True,
        ),
    ],
)
def test_betabinomial_vs_scipy(params, sp_params, skip_mode):
    """Test BetaBinomial distribution against scipy."""
    n_param = pt.constant(params[0], dtype="int64")
    alpha_param = pt.constant(params[1], dtype="float64")
    beta_param = pt.constant(params[2], dtype="float64")
    p_params = (n_param, alpha_param, beta_param)
    support = (0, int(np.max(params[0])))

    run_distribution_tests(
        p_dist=BetaBinomial,
        sp_dist=stats.betabinom,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        is_discrete=True,
        name="betabinomial",
        skip_mode=skip_mode,
    )
