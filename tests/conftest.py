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

jax.config.update("jax_enable_x64", True)

# after x64 is on, so nothing of DISCO-EB is built in single precision
from discoeb.component import Component  # noqa: E402
from discoeb.perturbations import PerturbationLayout  # noqa: E402


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

LAYOUTS = {
    "flat-lcdm": PerturbationLayout(),
    "dark-energy": PerturbationLayout(enable_dark_energy=True),
    "massive-nu": PerturbationLayout(nqmax=3, lmaxnu=8),
    # Blocks of unequal length: the hand-written Schur solver this replaced
    # could not take them.
    "ragged-blocks": PerturbationLayout(
        lmaxg=11, lmaxpol=13, lmaxr=17, nqmax=2, lmaxnu=9, enable_dark_energy=True
    ),
}


class ConstantThermalHistory(Component):
    """A thermal history frozen at one epoch, standing in for the recombination block.

    The defaults are a realistic epoch; a test that differences the
    right-hand side passes channels of order one, since at ``cs2 ~ 1e-8`` or
    ``a ~ 1e-3`` the baryon pressure and the dark-energy density fall below
    the rounding of the rest of the row.
    """

    def __init__(self, a=1e-3, opacity=50.0, cs2=1e-8, rhonu=3.0, pnu=0.5):
        self.a = lambda: a
        self.opacity = lambda: opacity
        self.cs2 = lambda: cs2
        self.rhonu = lambda: rhonu
        self.pnu = lambda: pnu


def pytest_sessionstart(session):
    jax.print_environment_info()
