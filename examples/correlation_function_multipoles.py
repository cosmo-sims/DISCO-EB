"""Compute the redshift-space correlation function multipoles and their derivatives.

Galaxy surveys measure clustering in redshift space, where peculiar
velocities squash structures along the line of sight. In linear theory
(Kaiser) a tracer of bias ``b`` has the power spectrum multipoles

    P_0 = (1 + 2/3 beta + 1/5 beta^2) b^2 P,
    P_2 = (4/3 beta + 4/7 beta^2) b^2 P,
    P_4 = 8/35 beta^2 b^2 P,

with ``beta = f / b`` and ``f = d ln delta / d ln a`` the growth rate. The
correlation function multipoles are their Hankel transforms,

    xi_l(s) = i^l / (2 pi^2) int dk k^2 P_l(k) j_l(k s).

``solve_eb`` keeps what this needs at the sample's redshift: the density
contrasts, their rates through the metric ``z`` and the baryon velocity
(``clxc' = -k z``, ``clxb' = -k (z + vb)``) and the expansion rate. The
scale factor is kept too, at a few times around the redshift, and an
interpolation in it gives every quantity at exactly that redshift for any
parameters. ``jax.jacfwd`` then gives the multipoles' derivatives, such as
the shift of the BAO peak with the Hubble constant.

Run from the repository root with the project's environment::

    uv run python examples/correlation_function_multipoles.py

It prints the multipoles and saves them and their derivatives beside this
file as ``correlation_function_multipoles.png``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from discoeb import LCDM, background_history, solve_eb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REDSHIFT = 0.5
BIAS = 2.0
SMOOTHING = 1.0  # Gaussian damping of P(k), Mpc: makes the transform converge
K = np.geomspace(1.0e-4, 4.0, 192)  # wave numbers of the solve, 1/Mpc
K_FINE = np.geomspace(K[0], K[-1], 8192)  # of the transform: resolves j_l(k s)
S = np.linspace(20.0, 200.0, 91)  # separations, Mpc

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
PARAMETERS = {"H0": "H_0", "Omegam": r"\Omega_m", "Omegab": r"\Omega_b", "n_s": "n_s"}
theta = jnp.asarray([getattr(planck, key) for key in PARAMETERS])

# Five save times around the redshift, from the fiducial background.
A_SAMPLE = 1.0 / (1.0 + REDSHIFT)
a_bg, tau_bg = background_history(planck)
TAU = np.interp(A_SAMPLE * np.linspace(0.9, 1.1, 5), a_bg, tau_bg)


def spherical_bessel(ell, x):
    """``j_l(x)`` for ``l`` of 0, 2 or 4: the closed form, and its series below 1/2."""

    big = jnp.maximum(x, 0.5)
    sin, cos = jnp.sin(big), jnp.cos(big)
    if ell == 0:
        closed, lead = sin / big, 1.0
    elif ell == 2:
        closed = (3.0 / big**3 - 1.0 / big) * sin - 3.0 / big**2 * cos
        lead = x**2 / 15.0
    else:
        closed = (105.0 / big**5 - 45.0 / big**3 + 1.0 / big) * sin
        closed -= (105.0 / big**4 - 10.0 / big**2) * cos
        lead = x**4 / 945.0
    n = 2 * ell + 3
    series = lead * (1.0 - x**2 / (2 * n) + x**4 / (8 * n * (n + 2)))
    return jnp.where(x < 0.5, series, closed)


def multipoles(theta):
    """``xi_0, xi_2, xi_4`` at the separations ``S``, ``(3, len(S))``."""

    cosmology = planck.replace(**dict(zip(PARAMETERS, theta)))
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

    beta = growth_rate / BIAS
    kaiser = (
        1.0 + 2.0 / 3.0 * beta + beta**2 / 5.0,
        4.0 / 3.0 * beta + 4.0 / 7.0 * beta**2,
        8.0 / 35.0 * beta**2,
    )
    xi = []
    for ell, sign, factor in zip((0, 2, 4), (1.0, -1.0, 1.0), kaiser):
        log_p = jnp.interp(np.log(K_FINE), np.log(K), jnp.log(BIAS**2 * factor * pk))
        integrand = K_FINE**3 * jnp.exp(log_p - (K_FINE * SMOOTHING) ** 2)
        bessel = spherical_bessel(ell, K_FINE * S[:, None])
        integral = jnp.trapezoid(integrand * bessel, np.log(K_FINE), axis=-1)
        xi.append(sign * integral / (2.0 * jnp.pi**2))
    return jnp.stack(xi)


xi = np.asarray(multipoles(theta))
dxi = np.asarray(jax.jacfwd(multipoles)(theta))  # (multipole, s, parameter)

print(f"z = {REDSHIFT}, b = {BIAS}")
print(f"{'s [Mpc]':>8}  {'s^2 xi_0':>10}  {'s^2 xi_2':>10}  {'s^2 xi_4':>10}")
for i in range(0, len(S), 10):
    print(f"{S[i]:8.0f}" + "".join(f"  {v:10.3f}" for v in S[i] ** 2 * xi[:, i]))
peak = (S > 120.0) & (S < 180.0)
print(f"BAO peak of xi_0 at s = {S[peak][np.argmax(xi[0, peak])]:.0f} Mpc")

fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharex=True)
styles = (("C0", r"$\xi_0$"), ("C2", r"$\xi_2$"), ("C4", r"$\xi_4$"))
for ell, (color, label) in enumerate(styles):
    axes[0].plot(S, S**2 * xi[ell], color=color, label=label)
    for i, ax in enumerate(axes[1:]):
        ax.plot(S, S**2 * theta[i] * dxi[ell, :, i], color=color)
axes[0].set_title(r"$s^2 \xi_\ell(s)$ [Mpc$^2$]")
axes[0].legend(frameon=False)
for ax, label in zip(axes[1:], PARAMETERS.values()):
    ax.set_title(rf"$s^2\, \partial \xi_\ell / \partial \log {label}$")
for ax in axes:
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"$s$ [Mpc]")
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)
