"""Time the matter power spectrum of 128 cosmologies at 128 wave numbers.

``solve_eb`` integrates every (cosmology, k) pair
in one GPU launch. The recombination history is a block of its own that
runs ahead of the perturbations on its own steps and hands down its opacity,
sound speed and scale factor as interpolants. The first call compiles the
kernel from scratch (about 35 s) and is timed on its own: the script points
numba's cache at an empty temporary directory, so modax's kernel cache is
neither read nor written;
the second call then uses a batch of cosmologies not seen before.

Run from the repository root with the project's environment::

    .venv/bin/python examples/matter_power_spectrum_batch.py
"""

import atexit
import os
import shutil
import tempfile

# Compile from fresh every run: an empty kernel cache, removed at exit. This
# must be set before numba or modax is imported.
os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="discoeb-kernel-cache-")
atexit.register(shutil.rmtree, os.environ["NUMBA_CACHE_DIR"], ignore_errors=True)

import time

import jax
import jax.numpy as jnp
import numpy as np

from discoeb import (
    LCDM,
    conformal_ages,
    matter_power_spectrum,
    solve_eb,
)


def cosmologies(n, offset):
    """Flat LCDM, omega_b, omega_c and h spread by under a per cent about Planck 2018."""
    out = []
    for phase in np.linspace(-1.0, 1.0, n) + offset:
        omega_b = 0.02238280 * (1.0 + 0.006 * phase)
        omega_c = 0.1201075 * (1.0 - 0.008 * phase)
        h = 0.6732117 * (1.0 + 0.004 * np.sin(np.pi * phase))
        out.append(
            LCDM(
                Omegam=(omega_b + omega_c) / h**2,
                Omegab=omega_b / h**2,
                A_s=2.100549e-9,
                n_s=0.9660499,
                H0=100.0 * h,
                Tcmb=2.7255,
                YHe=0.2454006,
                Neff=3.046,
            )
        )
    return out


def matter_power(k, batch):
    """``P(k)`` of a list of cosmologies: one ``solve_eb`` launch, then the spectrum from what it kept."""
    tau0 = conformal_ages(batch)
    solution = solve_eb(
        batch,
        k,
        tau0.max()[None],  # today: each trajectory stops at its own age
        keep=("clxc", "clxb", "z", "vb"),
        # recombination ten times looser than the hierarchy: moves P(k) by
        # under 6e-4 and takes 150 steps per trajectory instead of 225
        thermo_weight=0.1,
    )
    today = {name: solution[name][..., -1] for name in solution.names}
    params = {
        key: jnp.asarray([getattr(c, key) for c in batch])
        for key in ("Omegam", "Omegab", "A_s", "n_s", "k_p")
    }
    pk = matter_power_spectrum(k, **today, tau0=tau0, **params)
    return jax.block_until_ready(pk)


k = np.geomspace(1.0e-4, 1.0, 128)  # 1/Mpc

t0 = time.perf_counter()
matter_power(k, cosmologies(128, offset=0.0))
compile_time = time.perf_counter() - t0
print(f"compile and first call: {compile_time:.1f} s")

batch = cosmologies(128, offset=0.001)
t0 = time.perf_counter()
pk = matter_power(k, batch)
elapsed = time.perf_counter() - t0

print(f"P(k) for {pk.shape[0]} cosmologies x {pk.shape[1]} modes: {elapsed:.3f} s")
