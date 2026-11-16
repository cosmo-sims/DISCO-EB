"""Map the response of the redshift-space galaxy power spectrum to each parameter.

A survey observes the galaxy power spectrum as a function of the wave number
and of ``mu``, the cosine of the angle to the line of sight. Two effects make
it anisotropic. Peculiar velocities add the Kaiser factor,

    P_g(k, mu) = (b + f mu^2)^2 P(k),

and the survey turns redshifts and angles into distances with a reference
cosmology, which rescales the true wave numbers (Alcock-Paczynski):

    P_obs(k_ref, mu_ref) = P_g(k, mu) / (q_perp^2 q_par),
    q_perp = D_A / D_A,ref,   q_par = H_ref / H.

Everything comes from one ``solve_eb``: the density contrasts and their rates
give ``P`` and the growth rate ``f``, the expansion rate gives ``H``, and the
conformal time at the sample's scale factor gives the distance. ``jax.jacfwd``
then maps ``d ln P_obs / d ln p`` over the ``(k, mu)`` plane, one polar "eye"
per parameter with the line of sight pointing up.

Run from the repository root with the project's environment::

    uv run python examples/redshift_space_power_response.py

It prints the response across and along the line of sight and saves the maps
beside this file as ``redshift_space_power_response.png``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from discoeb import LCDM, background_history, conformal_ages, solve_eb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REDSHIFT = 1.0
K = np.geomspace(1.0e-4, 1.0, 128)  # wave numbers of the solve, 1/Mpc
K_REF = K[4:-4]  # of the map: the rescaled ones stay inside the solve's
MU = np.linspace(0.0, 1.0, 65)

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
PARAMETERS = {  # key: label; the last is the galaxy bias
    "H0": "H_0",
    "Omegam": r"\Omega_m",
    "Omegab": r"\Omega_b",
    "Neff": r"N_\mathrm{eff}",
    "n_s": "n_s",
    "bias": "b",
}
theta = jnp.asarray([getattr(planck, key) for key in list(PARAMETERS)[:-1]] + [1.5])

# Five save times around the redshift, from the reference background.
A_SAMPLE = 1.0 / (1.0 + REDSHIFT)
a_bg, tau_bg = background_history(planck)
TAU = np.interp(A_SAMPLE * np.linspace(0.9, 1.1, 5), a_bg, tau_bg)


def linear_theory(cosmology):
    """``P(k)``, the growth rate ``f(k)``, the comoving distance and ``a H`` at the redshift."""

    values = solve_eb(
        cosmology, K, TAU, keep=("clxc", "clxb", "z", "vb", "adotoa", "a")
    )
    # from the save times to the sample's scale factor: each (k,)
    a = values[0, -1, 0]
    clxc, clxb, z, vb, adotoa, _ = jax.vmap(
        jax.vmap(lambda v: jnp.interp(A_SAMPLE, a, v))
    )(values[0])

    f_b = cosmology.Omegab / cosmology.Omegam
    delta = (1.0 - f_b) * clxc + f_b * clxb
    growth_rate = -K * (z + f_b * vb) / (adotoa * delta)
    primordial = cosmology.A_s * (K / cosmology.k_p) ** (cosmology.n_s - 1.0)
    pk = 2.0 * jnp.pi**2 / K**3 * primordial * delta**2
    distance = conformal_ages(cosmology)[0] - jnp.interp(A_SAMPLE, a, TAU)
    return pk, growth_rate, distance, adotoa[0]


_, _, distance_ref, aH_ref = linear_theory(planck)


def observed_power(theta):
    """``P_obs(k_ref, mu_ref)`` in Mpc^3, ``(len(K_REF), len(MU))``."""

    *cosmological, bias = theta
    pk, growth_rate, distance, aH = linear_theory(
        planck.replace(**dict(zip(PARAMETERS, cosmological)))
    )
    q_perp, q_par = distance / distance_ref, aH_ref / aH
    stretch = jnp.sqrt(1.0 + MU**2 * ((q_perp / q_par) ** 2 - 1.0))
    k = K_REF[:, None] / q_perp * stretch  # the true wave number and angle
    mu = MU * q_perp / q_par / stretch
    pk = jnp.exp(jnp.interp(jnp.log(k), np.log(K), jnp.log(pk)))
    growth_rate = jnp.interp(jnp.log(k), np.log(K), growth_rate)
    return (bias + growth_rate * mu**2) ** 2 * pk / (q_perp**2 * q_par)


power = observed_power(theta)
jacobian = jax.jacfwd(observed_power)(theta)  # (k, mu, parameter)
response = np.asarray(theta * jacobian / power[..., None])  # d ln P / d ln p

print(f"d ln P_obs / d ln p at z = {REDSHIFT}, across (mu = 0) and along (mu = 1)")
print(f"{'k [1/Mpc]':>10}  {'mu':>3}" + "".join(f"{key:>9}" for key in PARAMETERS))
for i in range(6, len(K_REF), 22):
    for j in (0, -1):
        row = "".join(f"{v:9.3f}" for v in response[i, j])
        print(f"{K_REF[i]:10.3e}  {MU[j]:3.0f}{row}")

# One eye per parameter: the angle to the line of sight around, log k outwards.
angle = np.arccos(MU)
angle = np.concatenate(
    [angle, np.pi - angle[::-1], np.pi + angle, 2 * np.pi - angle[::-1]]
)
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), subplot_kw={"projection": "polar"})
for i, (ax, label) in enumerate(zip(axes.flat, PARAMETERS.values())):
    quadrant = response[..., i]
    eye = np.concatenate([quadrant, quadrant[:, ::-1], quadrant, quadrant[:, ::-1]], 1)
    centre = np.median(quadrant)
    limit = 3.0 * np.std(quadrant) + 1.0e-3
    mesh = ax.pcolormesh(
        angle,
        np.log10(K_REF),
        eye,
        cmap="RdBu_r",
        vmin=centre - limit,
        vmax=centre + limit,
        shading="gouraud",
    )
    fig.colorbar(mesh, ax=ax, shrink=0.6, pad=0.1)
    ax.set_theta_zero_location("N")
    ax.set_xticklabels([])
    ax.set_rticks([-3, -2, -1])
    ax.set_yticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$ / Mpc"], size=7)
    ax.set_title(rf"$\partial \log P(k, \mu) / \partial \log {label}$")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)
