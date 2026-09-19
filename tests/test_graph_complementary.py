"""Test if complementary functions share subexpressions when compiled together."""

import importlib

import pytest
from pytensor.compile.mode import Mode

from tests.helper_graph import compile_fn, compute_nodes, get_params, make_inputs

FAST_COMPILE_MODE = Mode(linker="py", optimizer="fast_compile")

DISTRIBUTIONS = [
    "asymmetriclaplace",
    "beta",
    "betascaled",
    "betaprime",
    "cauchy",
    "chisquared",
    "exgaussian",
    "exponential",
    "frechet",
    "gamma",
    "gumbel",
    "halfcauchy",
    "halfnormal",
    "halfstudentt",
    "inversegamma",
    "kumaraswamy",
    "laplace",
    "logistic",
    "logitnormal",
    "loglogistic",
    "lognormal",
    "moyal",
    "normal",
    "pareto",
    "rice",
    "skewnormal",
    "skew_studentt",
    "studentt",
    "triangular",
    "truncatednormal",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "bernoulli",
    "betabinomial",
    "binomial",
    "categorical",
    "discreteuniform",
    "discreteweibull",
    "geometric",
    "negativebinomial",
    "poisson",
    "zi_binomial",
    "zi_negativebinomial",
    "zi_poisson",
]

COMPLEMENTARY_PAIRS = [
    ("logcdf", "logsf"),
    ("cdf", "sf"),
    ("ppf", "isf"),
]


@pytest.mark.parametrize("module_name", DISTRIBUTIONS)
@pytest.mark.parametrize("fn1_name,fn2_name", COMPLEMENTARY_PAIRS)
def test_complementary_functions_share_subexpressions(module_name, fn1_name, fn2_name):
    """Complementary functions should share computation when compiled together.

    When both functions in a complementary pair (e.g., logcdf/logsf) are used
    in the same graph, they should share common subexpressions like the
    standardized value z = (x - mu) / sigma.

    If one function delegates to the other with transformed inputs (e.g.,
    logsf = logcdf(-x, -mu, sigma)), this sharing breaks and the combined
    graph becomes larger than necessary.
    """
    module = importlib.import_module(f"pytensor_distributions.{module_name}")

    if not hasattr(module, fn1_name) or not hasattr(module, fn2_name):
        pytest.skip(f"{module_name} missing {fn1_name} or {fn2_name}")

    params = get_params(module, fn1_name)
    inputs = make_inputs(params)

    fn1 = getattr(module, fn1_name)
    fn2 = getattr(module, fn2_name)

    fn1_only = compile_fn([fn1(*inputs)], inputs, mode=FAST_COMPILE_MODE)
    fn2_only = compile_fn([fn2(*inputs)], inputs, mode=FAST_COMPILE_MODE)
    combined = compile_fn([fn1(*inputs), fn2(*inputs)], inputs, mode=FAST_COMPILE_MODE)

    n_fn1 = len(compute_nodes(fn1_only))
    n_fn2 = len(compute_nodes(fn2_only))
    n_combined = len(compute_nodes(combined))

    min_shared = 1
    assert n_combined <= n_fn1 + n_fn2 - min_shared, (
        f"{module_name}.{fn1_name}/{fn2_name}: no subexpression sharing. "
        f"{fn1_name}={n_fn1}, {fn2_name}={n_fn2}, combined={n_combined}. "
        f"Expected combined < {n_fn1 + n_fn2 - min_shared}. "
        f"This likely means {fn2_name} is implemented via delegation to {fn1_name} "
    )
