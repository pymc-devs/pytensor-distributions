"""Test Wald (Inverse Gaussian) distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import wald as Wald
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 10.0], {"mu": 2.0 / 10.0, "scale": 10.0}),
        (
            [
                np.array([1.0, 3.0, 1.0, 10.0]),
                np.array([1.0, 5.0, 0.1, 100.0]),
            ],
            {
                "mu": np.array([1.0, 0.6, 10.0, 0.1]),
                "scale": np.array([1.0, 5.0, 0.1, 100.0]),
            },
        ),
    ],
)
def test_wald_vs_scipy(params, sp_params):
    """Test Wald distribution against scipy.stats.invgauss."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=Wald,
        sp_dist=stats.invgauss,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="wald",
        # Slightly higher tolerance for SF/logCDF due to numerical precision
        # when CDF is very close to 1 (error is ~1e-9 absolute, just over 1e-6 relative)
        sf_rtol=1.1e-6,
        logcdf_rtol=1.1e-6,
    )
