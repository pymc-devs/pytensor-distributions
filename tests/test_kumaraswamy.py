"""Test Kumaraswamy distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import kumaraswamy as Kumaraswamy
from tests.helper_empirical import run_empirical_tests
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([1.00000001, 5.0], {"a": 1.00000001, "b": 5}),
        (
            [
                np.array([1.00000001]),
                np.array([100.0]),
            ],
            {
                "a": np.array([1.00000001]),
                "b": np.array([100.0]),
            },
        ),
    ],
)
def test_kumaraswamy_vs_scipy(params, sp_params):
    """Test Kumaraswamy distribution against scipy beta (similar distribution)."""
    p_params = make_params(*params, dtype="float64")
    support = (0, 1)

    run_distribution_tests(
        p_dist=Kumaraswamy,
        sp_dist=stats.beta,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="kumaraswamy",
        use_quantiles_for_rvs=True,
    )


@pytest.mark.parametrize(
    "params",
    [
        [2.0, 3.0],
        [0.5, 3.0],
        [100.0, 100.0],
        [1.0, 1.0],
    ],
)
def test_kumaraswamy_vs_random(params):
    """Test Kumaraswamy distribution against random samples."""
    p_params = make_params(*params, dtype="float64")
    support = (0, 1)

    run_empirical_tests(
        p_dist=Kumaraswamy,
        p_params=p_params,
        support=support,
        name="kumaraswamy",
        sample_size=500_000,
        mean_rtol=1e-2,
        var_rtol=1e-2,
        std_rtol=1e-2,
        quantiles_rtol=1e-2,
        cdf_rtol=5e-2,
    )
