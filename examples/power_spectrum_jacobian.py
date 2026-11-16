"""Map how the matter power spectrum responds to every parameter, in scale and time.

``jax.jacfwd`` of one function gives the Jacobian of ``P(k, a)`` with respect
to ten parameters of w0waCDM with a massive neutrino. The eight that enter
the Einstein-Boltzmann solve share one forward-sensitivity launch. ``A_s``
and ``n_s`` only enter the primordial spectrum, so they cost nothing.

``solve_eb`` records at conformal times, and a derivative of its values holds
those times fixed. A response at fixed scale factor needs the scale factor
kept beside the density contrasts, and an interpolation onto the wanted
``a``. JAX then carries the shift of ``a(tau)`` with the parameters.

Run from the repository root with the project's environment::

    uv run python examples/power_spectrum_jacobian.py

It prints the response today and saves the maps beside this file as
``power_spectrum_jacobian.png``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from discoeb import W0WACDM_MNU, background_history, solve_eb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K = np.geomspace(1.0e-4, 1.0, 64)  # wave numbers, 1/Mpc
A_OUT = np.geomspace(0.01, 1.0, 48)  # scale factors of the map

fiducial = W0WACDM_MNU(
    Omegam=0.3099,
    Omegab=0.0488911,
    H0=67.742,
    A_s=2.1064e-9,
    n_s=0.96822,
    Tcmb=2.7255,
    YHe=0.248,
    Neff=2.046,  # the massless flavours
    Nmnu=1,
    mnu=0.06,  # eV
    w_DE_0=-0.99,
    w_DE_a=0.0,
    cs2_DE=1.0,
)
PARAMETERS = {  # key: label
    "H0": "H_0",
    "Omegam": r"\Omega_m",
    "Omegab": r"\Omega_b",
    "Neff": r"N_\mathrm{eff}",
    "mnu": r"m_\nu",
    "Tcmb": r"T_\mathrm{CMB}",
    "A_s": "A_s",
    "n_s": "n_s",
    "w_DE_0": "w_0",
    "w_DE_a": "w_a",
}
theta = jnp.asarray([getattr(fiducial, key) for key in PARAMETERS])

# Save times from before the first wanted scale factor until today.
a_bg, tau_bg = background_history(fiducial)
TAU = np.geomspace(np.interp(0.9 * A_OUT[0], a_bg, tau_bg), tau_bg[-1], 128)


def power(theta):
    """``P(k, a)`` of CDM and baryons in Mpc^3, ``(len(K), len(A_OUT))``."""

    cosmology = fiducial.replace(**dict(zip(PARAMETERS, theta)))
    values = solve_eb(cosmology, K, TAU, keep=("clxc", "clxb", "a"))
    clxc, clxb, a = values[0]  # each (k, tau)
    Omegam, Omegab = cosmology.Omegam, cosmology.Omegab
    delta = ((Omegam - Omegab) * clxc + Omegab * clxb) / Omegam
    # from the save times to the wanted scale factors, mode by mode
    delta = jax.vmap(lambda d: jnp.interp(A_OUT, a[0], d))(delta)
    k = K[:, None]
    primordial = cosmology.A_s * (k / cosmology.k_p) ** (cosmology.n_s - 1.0)
    return 2.0 * jnp.pi**2 / k**3 * primordial * delta**2


pk = power(theta)
jacobian = jax.jacfwd(power)(theta)  # (k, a, parameter)
# d ln P / d ln p, or d ln P / d p for a parameter whose fiducial value is zero
scale = jnp.where(theta == 0.0, 1.0, theta)
response = np.asarray(scale * jacobian / pk[..., None])

print("d ln P(k) / d ln p today (d ln P / d p for w_DE_a)")
print(f"{'k [1/Mpc]':>10}" + "".join(f"{key:>9}" for key in PARAMETERS))
for i in range(0, len(K), 9):
    print(f"{K[i]:10.3e}" + "".join(f"{v:9.3f}" for v in response[i, -1]))

fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True, sharey=True)
for i, (ax, label) in enumerate(zip(axes.flat, PARAMETERS.values())):
    limit = np.abs(response[..., i]).max()
    mesh = ax.pcolormesh(
        K, A_OUT, response[..., i].T, cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    fig.colorbar(mesh, ax=ax)
    log = "" if theta[i] == 0.0 else r"\log "
    ax.set_title(rf"$\partial \log P / \partial {log}{label}$")
    ax.loglog()
for ax in axes[-1]:
    ax.set_xlabel(r"$k$ [1/Mpc]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$a$")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)
