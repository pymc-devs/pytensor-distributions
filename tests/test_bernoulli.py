"""Test Bernoulli distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import bernoulli as Bernoulli
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.3], {"p": 0.3}),
        (
            [np.array([0.5, 0.001, 0.999])],
            {"p": np.array([0.5, 0.001, 0.999])},
        ),
    ],
)
def test_bernoulli_vs_scipy(params, sp_params):
    """Test Bernoulli distribution against scipy.stats.bernoulli."""
    p_params = make_params(*params)
    support = (0, 1)

    run_distribution_tests(
        p_dist=Bernoulli,
        sp_dist=stats.bernoulli,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="bernoulli",
        is_discrete=True,
    )
