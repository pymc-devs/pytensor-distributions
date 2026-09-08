"""Test Binomial distribution against scipy implementation."""

import numpy as np
import pytensor.tensor as pt
import pytest
from scipy import stats

from pytensor_distributions import binomial as Binomial
from tests.helper_scipy import run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params, logsf_rtol",
    [
        ([4, 0.4], {"n": 4, "p": 0.4}, 1e-6),
        (
            [
                np.array([10, 100, 20], dtype="int64"),
                np.array([0.3, 0.01, 0.8]),
            ],
            {
                "n": np.array([10, 100, 20]),
                "p": np.array([0.3, 0.01, 0.8]),
            },
            1e-2,
        ),
    ],
)
def test_binomial_vs_scipy(params, sp_params, logsf_rtol):
    """Test Binomial distribution against scipy.stats.binom."""
    n_param = pt.constant(params[0], dtype="int64")
    p_param = pt.constant(params[1], dtype="float64")
    p_params = (n_param, p_param)
    support = (0, int(np.max(params[0])))

    run_distribution_tests(
        p_dist=Binomial,
        sp_dist=stats.binom,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="binomial",
        is_discrete=True,
        logsf_rtol=logsf_rtol,
    )
