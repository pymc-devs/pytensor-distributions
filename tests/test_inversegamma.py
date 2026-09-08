"""Test InverseGamma distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import inversegamma as InverseGamma
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([5.0, 2.0], {"a": 5.0, "scale": 2.0}),
        (
            [
                np.array([0.1, 100.0]),
                np.array([0.1, 50.0]),
            ],
            {
                "a": np.array([0.1, 100.0]),
                "scale": np.array([0.1, 50.0]),
            },
        ),
    ],
)
def test_inversegamma_vs_scipy(params, sp_params):
    """Test InverseGamma distribution against scipy.stats.invgamma."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=InverseGamma,
        sp_dist=stats.invgamma,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="inversegamma",
        use_quantiles_for_rvs=True,
        quantiles_sample_size=200_000,
    )
