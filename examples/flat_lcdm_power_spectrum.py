"""Compute and time the linear matter power spectrum of 128 flat LCDM cosmologies.

An entry point to DISCO-EB. One ``jax.jit``-ed function makes the batch of
cosmologies, solves the Einstein-Boltzmann system of every (cosmology, k)
pair in one GPU launch with ``solve_eb`` and turns what it kept into ``P(k)``
with ``matter_power_spectrum``. Every parameter is traced, so a new batch
reuses the compiled function.

The first call traces the function and loads the GPU kernel, which modax
compiles once (about 35 s) and then caches on disk. The second call is timed
on a batch not seen before, as in an inference loop.

Run from the repository root with the project's environment::

    uv run python examples/flat_lcdm_power_spectrum.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from discoeb import LCDM, conformal_ages, matter_power_spectrum, solve_eb

N_COSMOLOGIES = 128
K = np.geomspace(1.0e-4, 1.0, 128)  # wave numbers, 1/Mpc


@jax.jit
def matter_power(offset: jax.Array) -> jax.Array:
    """``P(k)`` in Mpc^3 today, ``(N_COSMOLOGIES, len(K))``, of a batch about Planck 2018.

    The batch spreads ``omega_b``, ``omega_c`` and ``h`` by under a per cent
    about the Planck 2018 best fit (flat, three massless neutrino flavours).
    ``offset`` shifts the spread, so each value gives a different batch.
    """

    phase = jnp.linspace(-1.0, 1.0, N_COSMOLOGIES) + offset
    omega_b = 0.02238280 * (1.0 + 0.006 * phase)
    omega_c = 0.1201075 * (1.0 - 0.008 * phase)
    h = 0.6732117 * (1.0 + 0.004 * jnp.sin(jnp.pi * phase))
    cosmologies = [
        LCDM(
            Omegam=(omega_b[i] + omega_c[i]) / h[i] ** 2,
            Omegab=omega_b[i] / h[i] ** 2,
            H0=100.0 * h[i],
            A_s=2.100549e-9,
            n_s=0.9660499,
            k_p=0.05,  # pivot scale of A_s, 1/Mpc
            Tcmb=2.7255,
            YHe=0.2454006,
            Neff=3.046,
        )
        for i in range(N_COSMOLOGIES)
    ]

    tau0 = conformal_ages(cosmologies)
    solution = solve_eb(
        cosmologies,
        K,
        tau0.max()[None],  # today: each trajectory stops at its own age
        keep=("clxc", "clxb", "z", "vb"),
        # recombination ten times looser than the hierarchy: moves P(k) by
        # under 6e-4 and takes 150 steps per trajectory instead of 225
        thermo_weight=0.1,
    )
    today = {name: solution[name][..., -1] for name in solution.names}
    parameters = {
        key: jnp.stack([jnp.asarray(getattr(c, key)) for c in cosmologies])
        for key in ("Omegam", "Omegab", "A_s", "n_s", "k_p")
    }
    return matter_power_spectrum(K, **today, tau0=tau0, **parameters)


def timed(offset: float) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    pk = jax.block_until_ready(matter_power(jnp.asarray(offset)))
    return np.asarray(pk), time.perf_counter() - start


_, first = timed(0.0)
pk, elapsed = timed(0.001)  # a batch not seen before

print(f"first call, with tracing and the kernel's compile or load: {first:.1f} s")
print(f"P(k) of {pk.shape[0]} new cosmologies x {pk.shape[1]} modes: {elapsed:.3f} s")
print()
print("the central cosmology of the batch:")
print(f"{'k [1/Mpc]':>12}  {'P(k) [Mpc^3]':>14}")
central = pk[N_COSMOLOGIES // 2]
for kv, pv in zip(K[::8], central[::8]):
    print(f"{kv:12.4e}  {pv:14.6e}")
print(f"P(k) peaks at k = {K[np.argmax(central)]:.4g} 1/Mpc")
