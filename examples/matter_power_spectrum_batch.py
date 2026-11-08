"""Time the matter power spectrum of 128 cosmologies at 128 wave numbers.

``solve_matter_power_spectrum_batch`` integrates every (cosmology, k) pair
in one GPU launch. The recombination history is a block of its own that
runs ahead of the perturbations on its own steps and hands down its opacity,
sound speed and scale factor as interpolants. The first call compiles the
kernel (about 35 s) or loads it from modax's kernel cache (about 5 s), and
is timed on its own;
the second call then uses a batch of cosmologies not seen before.

Run from the repository root with the project's environment. Until modax
0.0.8, which has the Block and Hierarchy API, is released, put the local
modax checkout on the path::

    PYTHONPATH=$HOME/git/modax:$HOME/git/DISCO-EB/src .venv/bin/python examples/matter_power_spectrum_batch.py
"""

import os

# The perturbation kernel keeps its own device memory beside JAX's pool.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from discoeb.perturbations import solve_matter_power_spectrum_batch


def cosmologies(n, offset):
    """Flat LCDM, omega_b, omega_c and h spread by under a per cent about Planck 2018."""
    out = []
    for phase in np.linspace(-1.0, 1.0, n) + offset:
        omega_b = 0.02238280 * (1.0 + 0.006 * phase)
        omega_c = 0.1201075 * (1.0 - 0.008 * phase)
        h = 0.6732117 * (1.0 + 0.004 * np.sin(np.pi * phase))
        out.append(
            {
                "Omegam": (omega_b + omega_c) / h**2,
                "Omegab": omega_b / h**2,
                "w_DE_0": -1.0,
                "w_DE_a": 0.0,
                "cs2_DE": 1.0,
                "Omegak": 0.0,
                "A_s": 2.100549e-9,
                "n_s": 0.9660499,
                "H0": 100.0 * h,
                "Tcmb": 2.7255,
                "YHe": 0.2454006,
                "Neff": 3.046,
                "Nmnu": 0.0,
                "mnu": 0.0,
                "k_p": 0.05,
            }
        )
    return out


k = np.geomspace(1.0e-4, 1.0, 128)  # 1/Mpc

t0 = time.perf_counter()
solve_matter_power_spectrum_batch(k, cosmologies(128, offset=0.0))
compile_time = time.perf_counter() - t0
print(f"first call (compile, or load from the kernel cache): {compile_time:.1f} s")

batch = cosmologies(128, offset=0.001)
t0 = time.perf_counter()
pk = solve_matter_power_spectrum_batch(k, batch)
elapsed = time.perf_counter() - t0

print(f"P(k) for {pk.shape[0]} cosmologies x {pk.shape[1]} modes: {elapsed:.3f} s")
