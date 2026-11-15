"""Compute the linear matter power spectrum of a flat LCDM cosmology.

An entry point to DISCO-EB: apply a model to a parameter dictionary, solve the
Einstein-Boltzmann system with ``solve_eb`` and turn what it kept into ``P(k)``
with ``matter_power_spectrum``, all in one ``jax.jit``-ed function that
runs on the GPU. Every parameter is traced, so a new cosmology reuses the
compiled function. The first run compiles the GPU kernel (about 35 s); modax
caches it on disk, so later runs start in seconds.

Run from the repository root with the project's environment::

    .venv/bin/python examples/flat_lcdm_power_spectrum.py
"""

import jax
import numpy as np

from discoeb import (
    LCDM,
    conformal_ages,
    matter_power_spectrum,
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
    tau0 = conformal_ages(batch)[0]
    solution = solve_eb(
        batch,
        K,
        tau0[None],
        keep=("clxc", "clxb", "z", "vb"),
        # recombination ten times looser than the hierarchy: moves P(k) by
        # under 6e-4 and takes 150 steps per trajectory instead of 225
        thermo_weight=0.1,
    )
    today = {name: solution[name][0, :, -1] for name in solution.names}
    return matter_power_spectrum(
        K,
        **today,
        tau0=tau0,
        Omegam=cosmology["Omegam"],
        Omegab=cosmology["Omegab"],
        A_s=cosmology["A_s"],
        n_s=cosmology["n_s"],
        k_p=cosmology["k_p"],
    )


pk = np.asarray(solve_matter_power_spectrum(COSMOLOGY))

print(f"{'k [1/Mpc]':>12}  {'P(k) [Mpc^3]':>14}")
for kv, pv in zip(K[::4], pk[::4]):
    print(f"{kv:12.4e}  {pv:14.6e}")
print(f"P(k) peaks at k = {K[np.argmax(pk)]:.4g} 1/Mpc")
