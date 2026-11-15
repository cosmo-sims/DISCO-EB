"""Let JAX's GPU memory pool grow on demand, unless the user chose otherwise."""

import os
import warnings

# The perturbation solver is a numba-CUDA kernel launched beside JAX. The
# driver reserves its per-thread local memory at launch, outside JAX's pool,
# and with JAX preallocating 75% of the card the next XLA launch runs out of
# memory (CUDA error 2). JAX reads this when it creates the GPU client, at
# the first device operation, so the package imports this module before
# anything else. A memory setting of the user's own is left alone.
_MEMORY_VARIABLES = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")
if not any(name in os.environ for name in _MEMORY_VARIABLES):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        from jax._src.xla_bridge import backends_are_initialized
    except ImportError:  # a JAX that moved it: skip the check
        pass
    else:
        if backends_are_initialized():
            warnings.warn(
                "JAX initialised its backend before discoeb was imported, so its "
                "GPU memory pool is already preallocated and the perturbation "
                "kernel may run out of memory. Set "
                "XLA_PYTHON_CLIENT_PREALLOCATE=false before using JAX.",
                stacklevel=2,
            )
