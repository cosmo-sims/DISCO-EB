"""Compute the linear matter power spectrum of a flat LCDM cosmology.

An entry point to DISCO-EB: apply a model to a parameter dictionary, solve the
Einstein-Boltzmann system with ``solve_eb`` and turn what it kept into ``P(k)``
with ``compute_matter_power_spectrum``, all in one ``jax.jit``-ed function that
runs on the GPU. Every parameter is traced, so a new cosmology reuses the
compiled function. The first run compiles the GPU kernel (about 35 s); modax
caches it on disk, so later runs start in seconds.

Run from the repository root with the project's environment::

    .venv/bin/python examples/flat_lcdm_power_spectrum.py
"""

import os

# The perturbation kernel keeps its own device memory beside JAX's pool.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

# Before DISCO-EB builds anything: the solver works in double precision.
jax.config.update("jax_enable_x64", True)

import numpy as np

from discoeb import (
    LCDM,
    MATTER_POWER_QUANTITIES,
    MATTER_THERMO_WEIGHT,
    compute_matter_power_spectrum,
    conformal_ages,
    solve_eb,
)

# Planck 2018 best fit, flat, with three massless neutrino flavours.
COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "H0": 67.32117,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "k_p": 0.05,  # pivot scale of A_s, 1/Mpc
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
}

K = np.geomspace(1.0e-4, 1.0, 64)  # wave numbers, 1/Mpc


@jax.jit
def solve_matter_power_spectrum(cosmology: dict) -> jax.Array:
    """``P(k)`` in Mpc^3 of one cosmology today, at the wave numbers ``K``."""

    batch = LCDM([cosmology])
    solution = solve_eb(
        batch,
        K,
        conformal_ages(batch).max()[None],
        keep=MATTER_POWER_QUANTITIES,
        thermo_weight=MATTER_THERMO_WEIGHT,
    )
    return compute_matter_power_spectrum(solution, batch)[0]


pk = np.asarray(solve_matter_power_spectrum(COSMOLOGY))

print(f"{'k [1/Mpc]':>12}  {'P(k) [Mpc^3]':>14}")
for kv, pv in zip(K[::4], pk[::4]):
    print(f"{kv:12.4e}  {pv:14.6e}")
print(f"P(k) peaks at k = {K[np.argmax(pk)]:.4g} 1/Mpc")
