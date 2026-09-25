"""Shared utilities for graph quality and optimization tests."""

import importlib
import inspect

import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.compile.ops import DeepCopyOp
from pytensor.tensor.basic import Alloc
from pytensor.tensor.shape import Shape_i

# Ops that perform no per-element computation.
# A constant-valued expression legitimately compiles down to
# "read a shape, allocate a constant-filled array"
# instead of a single fused Composite
BOOKKEEPING_OPS = (Shape_i, Alloc, DeepCopyOp)


def compute_nodes(compiled_fn):
    """Apply nodes that perform actual computation, excluding shape bookkeeping."""
    return [
        node
        for node in compiled_fn.maker.fgraph.toposort()
        if not isinstance(node.op, BOOKKEEPING_OPS)
    ]


def compile_fn(outputs, inputs, mode="NUMBA"):
    """Compile a pytensor function with the given mode."""
    return pytensor.function(inputs, outputs, mode=mode, on_unused_input="ignore")


def skip_if_not_pure_elemwise(compiled_fn, module_name: str, function_name: str):
    """Skip test if the function contains non-pointwise operations."""
    from pytensor.tensor.elemwise import DimShuffle, Elemwise

    CLOSURE_OPS = (Elemwise, DimShuffle)

    nodes = list(compiled_fn.maker.fgraph.toposort())
    if not all(isinstance(node.op, CLOSURE_OPS) for node in nodes):
        non_closure = [
            type(node.op).__name__ for node in nodes if not isinstance(node.op, CLOSURE_OPS)
        ]
        pytest.skip(
            f"{module_name}.{function_name} contains non-pointwise op(s) {non_closure}; "
            "needs a dedicated, reviewed invariant instead of this generic check."
        )


def get_params(module, fn_name: str):
    """Get parameter names for a function in a module."""
    fn = getattr(module, fn_name)
    return list(inspect.signature(fn).parameters.keys())


def make_inputs(params: list[str]):
    """Create symbolic vector inputs for each parameter."""
    return [pt.dvector(p) for p in params]


def public_functions(module_name, excluded):
    """List every public, non-sampling callable defined (not merely imported) by a module."""
    full_module_name = f"pytensor_distributions.{module_name}"
    module = importlib.import_module(full_module_name)
    return sorted(
        name
        for name, member in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_")
        and name not in excluded
        and member.__module__ == full_module_name
    )


def array_inputs_for(fn):
    """Build one vector-valued symbolic input for every function parameter."""
    parameters = inspect.signature(fn).parameters.values()
    return [pt.dvector(parameter.name) for parameter in parameters]


def apply_nodes(compiled_fn):
    return list(compiled_fn.maker.fgraph.toposort())
