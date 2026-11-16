"""Differentiate the linear matter power spectrum with respect to Omega_m.

The smallest use of DISCO-EB's derivatives. ``power`` maps Omega_m to
``P(k)`` today through ``solve_eb`` and ``matter_power_spectrum``, and
``jax.jacfwd`` of it gives ``dP/dOmega_m`` at every wave number from one
forward-sensitivity launch. A central finite difference of two more solves
checks it.

Omega_b, H0 and flatness are held, so more matter means less dark energy.

Run from the repository root with the project's environment::

    uv run python examples/matter_power_derivative.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from discoeb import LCDM, conformal_ages, matter_power_spectrum, solve_eb

K = np.geomspace(1.0e-4, 1.0, 64)  # wave numbers, 1/Mpc

h = 0.6732117
planck = LCDM(
    Omegam=(0.02238280 + 0.1201075) / h**2,
    Omegab=0.02238280 / h**2,
    H0=100.0 * h,
    A_s=2.100549e-9,
    n_s=0.9660499,
    Tcmb=2.7255,
    YHe=0.2454006,
    Neff=3.046,
)


def power(Omegam):
    """``P(k)`` today in Mpc^3, ``(len(K),)``, of the Planck cosmology at this Omega_m."""

    cosmology = planck.replace(Omegam=Omegam)
    tau0 = conformal_ages(cosmology)[0]
    values = solve_eb(  # (cosmology, name, k, tau)
        cosmology, K, tau0.reshape(1), keep=("clxc", "clxb", "z", "vb")
    )
    clxc, clxb, z, vb = jnp.unstack(values[0, :, :, -1])
    return matter_power_spectrum(
        K,
        clxc=clxc,
        clxb=clxb,
        z=z,
        vb=vb,
        tau0=tau0,
        Omegam=cosmology.Omegam,
        Omegab=cosmology.Omegab,
        A_s=cosmology.A_s,
        n_s=cosmology.n_s,
        k_p=cosmology.k_p,
    )


Omegam = jnp.asarray(planck.Omegam)
pk = power(Omegam)
dpk = jax.jacfwd(power)(Omegam)
slope = np.asarray(Omegam * dpk / pk)  # d ln P / d ln Omega_m

step = 1.0e-3
finite = np.asarray(
    Omegam * jnp.log(power(Omegam + step) / power(Omegam - step)) / (2.0 * step)
)

print(
    f"{'k [1/Mpc]':>12}  {'P(k) [Mpc^3]':>14}  {'dlnP/dlnOm':>11}  {'finite diff.':>12}"
)
for i in range(0, len(K), 7):
    print(f"{K[i]:12.4e}  {pk[i]:14.6e}  {slope[i]:11.5f}  {finite[i]:12.5f}")
print(f"largest difference between the two slopes: {np.abs(slope - finite).max():.1e}")
