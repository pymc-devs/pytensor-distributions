"""Test AsymmetricLaplace distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import asymmetriclaplace as AsymmetricLaplace
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 1.0, 1.0], {"loc": 0.0, "scale": 1.0, "kappa": 1.0}),
        (
            [
                np.array([-1, 0, 5.0]),
                np.array([2, 1.0, 0.1]),
                np.array([2, 0.01, 100]),
            ],
            {
                "loc": np.array([-1, 0, 5.0]),
                "scale": np.array([2, 1.0, 0.1]),
                "kappa": np.array([2, 0.01, 100]),
            },
        ),
    ],
)
def test_asymmetriclaplace_vs_scipy(params, sp_params):
    """Test AsymmetricLaplace distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=AsymmetricLaplace,
        sp_dist=stats.laplace_asymmetric,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="asymmetriclaplace",
    )
