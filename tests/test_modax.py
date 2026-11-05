"""CUDA target selection without importing or initializing a GPU backend."""

import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from discoeb import _modax


@pytest.fixture
def backend(monkeypatch):
    _modax._configure_cuda_target.cache_clear()
    # Record even an initially absent variable, since setup writes it itself.
    monkeypatch.setenv("NUMBA_ENZYME_CUDA_CC", "")
    monkeypatch.delenv("NUMBA_ENZYME_CUDA_CC")
    versions = _modax._LLVM15_STACK.copy()
    monkeypatch.setattr(_modax, "version", versions.__getitem__)

    tools = SimpleNamespace(
        get_gpu_compute_capability=Mock(return_value=(12, 0)),
        _cached_cc=(12, 0),
    )
    compile_kernel = Mock(return_value=123)
    bridge = SimpleNamespace(compile_kernel=compile_kernel)
    solve = Mock(return_value="solution")
    target = Mock(return_value=(9, 0))
    modules = {
        "numba_cuda_mlir": {"tools": tools},
        "numba_enzyme": {},
        "numba_enzyme.cuda": {"_compute_capability": target},
        "modax": {"_jax_numba_custom_call": bridge},
        "modax.rodas5P": {"solve": solve},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    yield SimpleNamespace(
        tools=tools,
        bridge=bridge,
        compile_kernel=compile_kernel,
        solve=solve,
        target=target,
        versions=versions,
    )
    _modax._configure_cuda_target.cache_clear()


def kernel(module="modax.rodas5P"):
    return SimpleNamespace(
        py_func=SimpleNamespace(__module__=module),
        targetoptions={"chip": "sm_120", "max_registers": 168},
    )


def test_import_does_not_initialize_cuda(backend):
    spec = importlib.util.spec_from_file_location("_modax_import_test", _modax.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    backend.tools.get_gpu_compute_capability.assert_not_called()
    backend.target.assert_not_called()
    assert backend.bridge.compile_kernel is backend.compile_kernel
    assert "NUMBA_ENZYME_CUDA_CC" not in os.environ


def test_blackwell_uses_compatible_ir_and_keeps_physical_target(backend):
    assert _modax.solve_rodas5p("rhs", "initial", rtol=1e-4) == "solution"
    backend.solve.assert_called_once_with("rhs", "initial", rtol=1e-4)
    assert os.environ["NUMBA_ENZYME_CUDA_CC"] == "9.0"
    backend.target.assert_called_once_with(None, backend.tools)

    compiled = kernel()
    assert backend.bridge.compile_kernel(compiled, ("float64",)) == 123
    assert compiled.targetoptions == {"chip": "sm_90", "max_registers": 168}
    assert backend.tools._cached_cc == (12, 0)
    backend.compile_kernel.assert_called_once_with(compiled, ("float64",))


def test_explicit_enzyme_target_is_used_for_the_outer_kernel(backend, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CUDA_CC", "8.6")
    backend.target.return_value = (8, 6)
    _modax._configure_cuda_target()

    compiled = kernel()
    backend.bridge.compile_kernel(compiled, ())
    assert os.environ["NUMBA_ENZYME_CUDA_CC"] == "8.6"
    backend.target.assert_called_once_with(None, backend.tools)
    assert compiled.targetoptions["chip"] == "sm_86"


def test_invalid_explicit_target_keeps_enzyme_validation(backend, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CUDA_CC", "12.0")
    backend.target.side_effect = RuntimeError("unsupported compute capability")

    with pytest.raises(RuntimeError, match="unsupported compute capability"):
        _modax.solve_rodas5p("rhs", "initial")

    assert os.environ["NUMBA_ENZYME_CUDA_CC"] == "12.0"
    assert backend.bridge.compile_kernel is backend.compile_kernel
    backend.solve.assert_not_called()


def test_setup_is_idempotent_and_leaves_other_kernels_alone(backend):
    _modax._configure_cuda_target()
    wrapped = backend.bridge.compile_kernel
    _modax._configure_cuda_target()
    assert backend.bridge.compile_kernel is wrapped
    backend.tools.get_gpu_compute_capability.assert_called_once_with(tuple)

    compiled = kernel("modax.tsit5")
    wrapped(compiled, ())
    assert compiled.targetoptions["chip"] == "sm_120"
    backend.compile_kernel.assert_called_once_with(compiled, ())


@pytest.mark.parametrize("capability", [(7, 0), (8, 9), (9, 0)])
def test_supported_gpus_are_unchanged(backend, capability):
    backend.tools.get_gpu_compute_capability.return_value = capability
    _modax._configure_cuda_target()

    assert backend.bridge.compile_kernel is backend.compile_kernel
    backend.target.assert_not_called()
    assert "NUMBA_ENZYME_CUDA_CC" not in os.environ


@pytest.mark.parametrize("package", _modax._LLVM15_STACK)
def test_newer_dependencies_are_unchanged(backend, package):
    backend.versions[package] = "99.0.0"
    _modax._configure_cuda_target()

    backend.tools.get_gpu_compute_capability.assert_not_called()
    assert backend.bridge.compile_kernel is backend.compile_kernel
    assert "NUMBA_ENZYME_CUDA_CC" not in os.environ
