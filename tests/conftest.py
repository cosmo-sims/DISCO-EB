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
import pytest

jax.config.update("jax_enable_x64", True)

# after x64 is on, so nothing of DISCO-EB is built in single precision
from helpers import _cuda_available  # noqa: E402


def pytest_configure(config):
    """Keep JAX on the CPU in a run without the ``gpu`` tests.

    Such a run then holds no device memory, in any xdist worker, and may
    share the machine with a run on the GPU.
    """

    if "not gpu" in (config.getoption("markexpr") or ""):
        jax.config.update("jax_platforms", "cpu")


def pytest_collection_modifyitems(config, items):
    """Keep the tests marked ``gpu`` on one worker, and skip them without a CUDA device.

    Under pytest-xdist (``-n auto``) the host tests spread over the cores.
    Two processes on one GPU return garbage and no error, so every ``gpu``
    test joins one group, which ``--dist loadgroup`` gives to one worker.
    """

    gpu = [item for item in items if item.get_closest_marker("gpu") is not None]
    for item in gpu:
        item.add_marker(pytest.mark.xdist_group("gpu"))
    if not _cuda_available():
        skip = pytest.mark.skip(reason="needs a CUDA GPU")
        for item in gpu:
            item.add_marker(skip)


def pytest_sessionstart(session):
    jax.print_environment_info()
