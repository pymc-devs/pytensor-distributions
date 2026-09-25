"""Test if expressions are written in a graph optimization-friendly way."""

import importlib

import pytest
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import NUMBA, Mode
from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.utils import MethodNotDefined
from pytensor.scalar.basic import Composite
from pytensor.tensor.elemwise import DimShuffle, Elemwise

from tests.helper_graph import (
    apply_nodes,
    array_inputs_for,
    compile_fn,
    compute_nodes,
    public_functions,
)

CLOSURE_OPS = (Elemwise, DimShuffle, OpFromGraph, DeepCopyOp)

UNOPTIMIZED_MODE = Mode(linker="py", optimizer=None)
OPTIMIZED_MODE = Mode(linker="py", optimizer=NUMBA.optimizer)

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

MODULE_FUNCTION_CASES = [
    (module_name, function_name)
    for module_name in DISTRIBUTIONS
    for function_name in public_functions(module_name, EXCLUDED_FUNCTIONS)
]


def _n_unfusable(compiled_fn):
    """Count Elemwise nodes whose scalar op has no C code and therefore cannot fuse."""
    count = 0
    for node in compiled_fn.maker.fgraph.toposort():
        sop = getattr(node.op, "scalar_op", None)
        if sop is None or isinstance(sop, Composite):
            continue
        try:
            sop.c_code(None, "n", ["x"] * 4, ["z"], {})
        except (NotImplementedError, MethodNotDefined):
            count += 1
        except Exception:
            pass
    return count


@pytest.mark.parametrize(("module_name", "function_name"), MODULE_FUNCTION_CASES)
def test_pure_elemwise_expressions_fuse_under_numba_rewrites(module_name, function_name):
    """A pure-Elemwise expression must fuse to at most 1 + 2*n_unfusable compute nodes."""
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
    max_compute_nodes = 1 + 2 * _n_unfusable(optimized)

    assert len(nodes) <= max_compute_nodes, (
        f"{module_name}.{function_name} optimized to {len(nodes)} compute "
        f"node(s), expected at most {max_compute_nodes}: "
        f"{[type(node.op).__name__ for node in nodes]}."
    )
