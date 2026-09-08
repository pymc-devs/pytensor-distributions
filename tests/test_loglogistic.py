import numpy as np
import pytest
from scipy import stats

from pytensor_distributions import loglogistic as LogLogistic
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([1.0, 5.0], {"c": 5.0, "scale": 1.0}),
        ([1.0, 1.5], {"c": 1.5, "scale": 1.0}),
        (
            [
                np.array([2.0, 0.5, 3.0]),
                np.array([3.0, 10.0, 2.5]),
            ],
            {
                "c": np.array([3.0, 10.0, 2.5]),
                "scale": np.array([2.0, 0.5, 3.0]),
            },
        ),
    ],
)
def test_loglogistic_vs_scipy(params, sp_params):
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=LogLogistic,
        sp_dist=stats.fisk,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="loglogistic",
    )
