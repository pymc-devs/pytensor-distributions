"""Test Chi-squared distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import chisquared as ChiSquared
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([1.0], {"df": 1.0}),
        ([np.array([5.0, 0.5, 100.0])], {"df": np.array([5.0, 0.5, 100.0])}),
    ],
)
def test_chisquared_vs_scipy(params, sp_params):
    """Test Chi-squared distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (0.0, float("inf"))

    run_distribution_tests(
        p_dist=ChiSquared,
        sp_dist=stats.chi2,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="chisquared",
    )
