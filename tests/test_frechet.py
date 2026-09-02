"""Test Fréchet distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import frechet as Frechet
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 5.0], {"c": 2.0, "scale": 5.0}),
        ([0.5, 3.0], {"c": 0.5, "scale": 3.0}),
        ([1.0, 1.0], {"c": 1.0, "scale": 1.0}),
        ([100.0, 2.0], {"c": 100.0, "scale": 2.0}),
    ],
)
def test_frechet_vs_scipy(params, sp_params):
    """Test Fréchet distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=Frechet,
        sp_dist=stats.invweibull,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="frechet",
        use_quantiles_for_rvs=True,
    )
