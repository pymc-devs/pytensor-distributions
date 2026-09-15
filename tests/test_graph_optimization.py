"""Test if expressions are written in a graph optimization-friendly way."""

import importlib

import pytest
from pytensor.compile.mode import Mode
from pytensor.tensor.elemwise import DimShuffle, Elemwise

from tests.helper_graph import (
    apply_nodes,
    array_inputs_for,
    compile_fn,
    compute_nodes,
    public_functions,
)

# DimShuffle commonly appears only to pad a Python scalar constant (e.g. ``0.5``)
# up to the input's rank; it is folded away by the rewriter, so it doesn't
# disqualify an expression from being a pure pointwise closure.
CLOSURE_OPS = (Elemwise, DimShuffle)

UNOPTIMIZED_MODE = Mode(linker="py", optimizer=None)
OPTIMIZED_MODE = "NUMBA"

DISTRIBUTIONS = [
    # Continuous
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
    # "polyagamma",
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
    # Discrete
    "bernoulli",
    "betabinomial",
    "binomial",
    "categorical",
    "discreteuniform",
    "discreteweibull",
    "geometric",
    # "hypergeometric",
    "negativebinomial",
    "poisson",
    "zi_binomial",
    "zi_negativebinomial",
    "zi_poisson",
]

EXCLUDED_FUNCTIONS = {"rvs", "expect"}

MAX_COMPUTE_NODES = {
    ("betaprime", "isf"): 2,
    ("betaprime", "median"): 2,
    ("betaprime", "ppf"): 2,
    ("beta", "isf"): 2,
    ("beta", "ppf"): 2,
    ("betascaled", "isf"): 2,
    ("betascaled", "median"): 2,
    ("betascaled", "ppf"): 2,
    ("chisquared", "isf"): 3,
    ("chisquared", "median"): 3,
    ("chisquared", "ppf"): 3,
    ("exgaussian", "mode"): 3,
    ("gamma", "isf"): 2,
    ("gamma", "median"): 2,
    ("gamma", "ppf"): 2,
    ("halfnormal", "isf"): 2,
    ("halfnormal", "ppf"): 2,
    ("halfstudentt", "isf"): 5,
    ("halfstudentt", "median"): 3,
    ("halfstudentt", "ppf"): 5,
    ("inversegamma", "isf"): 2,
    ("inversegamma", "median"): 2,
    ("inversegamma", "ppf"): 3,
    ("logitnormal", "isf"): 3,
    ("logitnormal", "ppf"): 3,
    ("lognormal", "isf"): 3,
    ("lognormal", "ppf"): 3,
    ("moyal", "isf"): 2,
    ("moyal", "ppf"): 3,
    ("normal", "isf"): 3,
    ("normal", "ppf"): 3,
    ("rice", "lmoment1"): 4,
    ("rice", "logpdf"): 3,
    ("rice", "mean"): 4,
    ("rice", "pdf"): 3,
    ("rice", "std"): 4,
    ("rice", "var"): 4,
    ("skew_studentt", "isf"): 2,
    ("skew_studentt", "median"): 2,
    ("skew_studentt", "ppf"): 2,
    ("skewnormal", "cdf"): 3,
    ("skewnormal", "logcdf"): 3,
    ("skewnormal", "logsf"): 3,
    ("skewnormal", "sf"): 3,
    ("studentt", "isf"): 5,
    ("studentt", "ppf"): 5,
    ("truncatednormal", "isf"): 4,
    ("truncatednormal", "median"): 4,
    ("truncatednormal", "ppf"): 4,
    ("vonmises", "entropy"): 3,
    ("vonmises", "logpdf"): 2,
    ("vonmises", "pdf"): 2,
    ("vonmises", "std"): 3,
    ("vonmises", "var"): 3,
}


MODULE_FUNCTION_CASES = [
    (module_name, function_name)
    for module_name in DISTRIBUTIONS
    for function_name in public_functions(module_name, EXCLUDED_FUNCTIONS)
]


@pytest.mark.parametrize(("module_name", "function_name"), MODULE_FUNCTION_CASES)
def test_pure_elemwise_expressions_fuse_to_one_composite_under_numba(module_name, function_name):
    """A pure-Elemwise expression must compile to a single fused op under Numba."""
    module = importlib.import_module(f"pytensor_distributions.{module_name}")
    fn = getattr(module, function_name)
    inputs = array_inputs_for(fn)
    out = fn(*inputs)

    unoptimized_nodes = apply_nodes(compile_fn(out, inputs, mode=UNOPTIMIZED_MODE))
    if not all(isinstance(node.op, CLOSURE_OPS) for node in unoptimized_nodes):
        non_closure = [
            type(node.op).__name__
            for node in unoptimized_nodes
            if not isinstance(node.op, CLOSURE_OPS)
        ]
        pytest.skip(
            f"{module_name}.{function_name} contains non-pointwise op(s) {non_closure}; "
            "needs a dedicated, reviewed invariant instead of this generic fusion check."
        )

    optimized = compile_fn(out, inputs, mode=OPTIMIZED_MODE)
    nodes = compute_nodes(optimized)
    max_compute_nodes = MAX_COMPUTE_NODES.get((module_name, function_name), 1)

    assert len(nodes) <= max_compute_nodes, (
        f"{module_name}.{function_name} optimized to {len(nodes)} compute "
        f"node(s), expected at most {max_compute_nodes}: "
        f"{[type(node.op).__name__ for node in nodes]}."
    )
