import os

# The perturbation solver is a numba-CUDA kernel launched beside JAX. The
# driver reserves its per-thread local memory (tens of KB a thread for a joint
# state-plus-sensitivity system, times every thread the device can hold) at
# launch, outside JAX's pool. With JAX preallocating 75% of the card up front
# the two together exhaust a 12 GB GPU and the next XLA launch fails with CUDA
# error 2. Let the JAX pool grow on demand instead, unless the environment
# already says otherwise.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from types import SimpleNamespace

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# after x64 is on, so nothing of DISCO-EB is built in single precision
from discoeb.component import Component  # noqa: E402
from discoeb.models import LCDM, LCDM_MNU, W0WACDM, W0WACDM_MNU  # noqa: E402


def _cuda_available() -> bool:
    """Whether numba-CUDA sees a GPU, for tests that launch a kernel directly."""

    try:
        from numba_cuda_mlir import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


# Host stand-ins for a build: an identity jit, numpy scratch arrays, float64.
HOST = dict(
    jit=lambda f: f,
    cuda=SimpleNamespace(
        local=SimpleNamespace(array=lambda shape, dtype: np.zeros(shape))
    ),
    float64=np.float64,
)

MODELS = {
    "flat-lcdm": LCDM,
    "dark-energy": W0WACDM,
    "massive-nu": LCDM_MNU.with_settings(NQMAX=3, LMAX_NU=8),
    # Blocks of unequal length: the hand-written Schur solver this replaced
    # could not take them.
    "ragged-blocks": W0WACDM_MNU.with_settings(
        LMAX_G=11, LMAX_POL=13, LMAX_NR=17, NQMAX=2, LMAX_NU=9
    ),
}


def cosmologies_of(model, params):
    """The cosmologies of parameter dictionaries: one for a dictionary, a list for a list.

    Through :meth:`Model.from_dict <discoeb.models.Model.from_dict>`, so a
    dictionary written for another model passes while its other keys keep
    their defaults.
    """

    if isinstance(params, dict):
        return model.from_dict(params)
    return [model.from_dict(p) for p in params]


def state_index(model) -> dict[str, int]:
    """The perturbation state's indices by element name."""

    return {name: i for i, name in enumerate(model.state_names())}


class ConstantThermalHistory(Component):
    """A thermal history frozen at one epoch, standing in for the recombination block.

    The defaults are a realistic epoch; a test that differences the
    right-hand side passes channels of order one, since at ``cs2 ~ 1e-8`` or
    ``a ~ 1e-3`` the baryon pressure and the dark-energy density fall below
    the rounding of the rest of the row.
    """

    def __init__(self, a=1e-3, opacity=50.0, cs2=1e-8):
        self.a = lambda: a
        self.opacity = lambda: opacity
        self.cs2 = lambda: cs2


GPU_MODULES = {"test_perturbations", "test_thermo", "test_reionization"}
"""Test modules whose every test launches the perturbation kernel."""


def pytest_collection_modifyitems(config, items):
    """Skip the kernel tests without a GPU, as on CI's fallback runner."""

    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if item.module is not None and item.module.__name__ in GPU_MODULES:
            item.add_marker(skip)


def pytest_sessionstart(session):
    jax.print_environment_info()
