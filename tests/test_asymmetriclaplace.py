"""Test AsymmetricLaplace distribution against scipy implementation."""

import numpy as np
import pytensor.tensor as pt
import pytest
from scipy import stats

from pytensor_distributions import asymmetriclaplace as AsymmetricLaplace
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 1.0, 1.0], {"loc": 0.0, "scale": 1.0, "kappa": 1.0}),
        ([-1.0, 2.0, 2.0], {"loc": -1.0, "scale": 2.0, "kappa": 2.0}),
        ([0.0, 1.0, 0.01], {"loc": 0.0, "scale": 1.0, "kappa": 0.01}),
        ([5.0, 0.1, 100.0], {"loc": 5.0, "scale": 0.1, "kappa": 100.0}),
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


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_median_symmetric_is_exact_and_dtype_preserving(dtype):
    """At kappa=1 (symmetric Laplace) the median is exactly the location, in any dtype.

    Guards the dtype-aware log(2) constant: a float64 constant would upcast float32
    inputs and leave a ~1.9e-9 residual instead of an exact 0.
    """
    mu, b, kappa = (pt.constant(v, dtype=dtype) for v in (3.0, 2.0, 1.0))
    median = AsymmetricLaplace.median(mu, b, kappa)

    assert median.dtype == dtype
    np.testing.assert_array_equal(median.eval(), np.asarray(3.0, dtype=dtype))
