import os

# The perturbation solver is a numba-CUDA kernel launched beside JAX. The
# driver reserves its per-thread local memory (tens of KB a thread for a joint
# state-plus-sensitivity system, times every thread the device can hold) at
# launch, outside JAX's pool. With JAX preallocating 75% of the card up front
# the two together exhaust a 12 GB GPU and the next XLA launch fails with CUDA
# error 2. Let the JAX pool grow on demand instead, unless the environment
# already says otherwise.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)


def _cuda_available() -> bool:
    """Whether numba-CUDA sees a GPU, for tests that launch a kernel directly."""

    try:
        from numba_cuda_mlir import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def pytest_sessionstart(session):
    jax.print_environment_info()
