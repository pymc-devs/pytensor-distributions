"""Test Triangular distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import triangular as Triangular
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 0.5, 1.0], {"loc": 0.0, "scale": 1.0, "c": 0.5}),
        (
            [
                np.array([-5.0, 0.0, -1e6, 1.0]),
                np.array([-4.0, 0.1, 0.0, 1.001]),
                np.array([-3.0, 1.0, 1e6, 1.004]),
            ],
            {
                "loc": np.array([-5.0, 0.0, -1e6, 1.0]),
                "scale": np.array([2.0, 1.0, 2e6, 0.004]),
                "c": np.array([0.5, 0.1, 0.5, 0.25]),
            },
        ),
    ],
)
def test_triangular_vs_scipy(params, sp_params):
    """Test Triangular distribution against scipy."""
    lower, _, upper = params
    p_params = make_params(*params, dtype="float64")
    support = (float(np.min(lower)), float(np.max(upper)))

    run_distribution_tests(
        p_dist=Triangular,
        sp_dist=stats.triang,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="triangular",
    )
