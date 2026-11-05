"""Lazy access to modax with compatibility for its LLVM 15 Enzyme backend."""

import os
from functools import cache, wraps
from importlib.metadata import version


_LLVM15_STACK = {
    "modax-solvers": "0.0.4",
    "numba-enzyme-cuda": "0.2.2",
    "numba-cuda-mlir": "0.5.1",
}


@cache
def _configure_cuda_target():
    """Use compatible IR for Blackwell, while linking for the actual GPU.

    This pinned stack's Enzyme plugin needs LLVM 15, whereas a native
    Blackwell target selects LLVM 20. Both the derivative and its calling
    Rodas5P kernel must use the older IR path. numba-cuda-mlir still links the
    result for the physical GPU when the requested chip is older.

    modax 0.0.4 has no public kernel-target option. Keep this private bridge
    hook limited to that stack and remove it when modax exposes one. The
    hook persists because JAX can compile sensitivity kernels after the
    original solve returns. GPU discovery happens only on the first solve.
    """
    if any(version(name) != expected for name, expected in _LLVM15_STACK.items()):
        return

    from numba_cuda_mlir import tools

    if tools.get_gpu_compute_capability(tuple) < (10, 0):
        return

    from modax import _jax_numba_custom_call as bridge
    from numba_enzyme.cuda import _compute_capability

    os.environ.setdefault("NUMBA_ENZYME_CUDA_CC", "9.0")
    # Use Enzyme's own parser and validation, including explicit user targets.
    major, minor = _compute_capability(None, tools)
    chip = f"sm_{major}{minor}"
    original_compile = bridge.compile_kernel

    @wraps(original_compile)
    def compile_kernel(kernel, argtypes):
        if kernel.py_func.__module__ == "modax.rodas5P":
            kernel.targetoptions["chip"] = chip
        return original_compile(kernel, argtypes)

    bridge.compile_kernel = compile_kernel


def solve_rodas5p(*args, **kwargs):
    """Call Rodas5P after lazily configuring the affected CUDA toolchain."""
    _configure_cuda_target()
    from modax.rodas5P import solve

    return solve(*args, **kwargs)
