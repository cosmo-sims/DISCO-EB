"""Forecast the parameter constraints of a Euclid-like spectroscopic galaxy survey.

A Fisher forecast needs the derivatives of the observable with respect to
every parameter. Here the observable is the galaxy power spectrum in four
redshift bins, with the Kaiser effect and the Alcock-Paczynski distortion of
a survey analysed in a reference cosmology:

    P_obs(k_ref, mu_ref) = (b + f mu^2)^2 P(k) / (q_perp^2 q_par),
    q_perp = D_A / D_A,ref,   q_par = H_ref / H.

One ``solve_eb`` gives the linear theory of every bin, and one ``jax.jacfwd``
its derivatives with respect to five cosmological parameters and the four
biases. The Fisher matrix sums the bins of volume ``V`` and galaxy density
``n``,

    F_ij = sum_bins V / (4 pi^2) int_0^1 dmu int dk k^2
           d_i ln P_obs  d_j ln P_obs  (n P_obs / (1 + n P_obs))^2,

and the inverse's diagonal gives the errors marginalised over the rest. The
survey numbers follow the Euclid forecast of arXiv:1910.09273. This minimal
model leaves out its non-linear damping, redshift errors and shot-noise
nuisance parameters.

Run from the repository root with the project's environment::

    uv run python examples/fisher_forecast.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from discoeb import LCDM, background_history, conformal_ages, solve_eb

h = 0.67
fiducial = LCDM(
    Omegam=0.32,
    Omegab=0.05,
    H0=100.0 * h,
    A_s=2.1064e-9,
    n_s=0.96,
    Tcmb=2.7255,
    YHe=0.248,
    Neff=3.046,
)

# The survey: 15000 square degrees in four redshift bins.
REDSHIFTS = np.array([1.0, 1.2, 1.4, 1.65])
BIASES = np.array([1.46, 1.61, 1.75, 1.90])
VOLUMES = np.array([7.94, 9.15, 10.05, 16.22]) * 1.0e9 / h**3  # Mpc^3
GALAXIES = np.array([1815.0, 1701.5, 1410.0, 940.97]) * 15000.0  # per unit redshift
DENSITIES = GALAXIES * np.array([0.2, 0.2, 0.2, 0.3]) / VOLUMES  # 1/Mpc^3

K = np.geomspace(2.0e-4, 0.5, 128)  # wave numbers of the solve, 1/Mpc
K_REF = np.geomspace(1.0e-3 * h, 0.25 * h, 96)  # those the survey uses
MU = np.linspace(0.0, 1.0, 33)

# The parameters: four of the cosmology, ln(10^10 A_s) and the bins' biases.
COSMOLOGICAL = ("Omegam", "Omegab", "H0", "n_s")
NAMES = (*COSMOLOGICAL, "ln10^10A_s", *(f"b(z={z})" for z in REDSHIFTS))
theta = jnp.asarray(
    [
        *(getattr(fiducial, key) for key in COSMOLOGICAL),
        np.log(1.0e10 * fiducial.A_s),
        *BIASES,
    ]
)

# Save times that cover the bins, from the reference background.
A_BINS = 1.0 / (1.0 + REDSHIFTS)
a_bg, tau_bg = background_history(fiducial)
TAU = np.interp(np.linspace(0.9 * A_BINS[-1], 1.1 * A_BINS[0], 12), a_bg, tau_bg)


def linear_theory(cosmology):
    """``P(k)`` and the growth rate ``f(k)``, each ``(bin, k)``, the distance and ``a H``, each ``(bin,)``."""

    values = solve_eb(
        cosmology, K, TAU, keep=("clxc", "clxb", "z", "vb", "adotoa", "a")
    )
    # from the save times to the bins' scale factors: each (k, bin)
    a = values[0, -1, 0]
    clxc, clxb, z, vb, adotoa, _ = jax.vmap(
        jax.vmap(lambda v: jnp.interp(A_BINS, a, v))
    )(values[0])

    k = K[:, None]
    f_b = cosmology.Omegab / cosmology.Omegam
    delta = (1.0 - f_b) * clxc + f_b * clxb
    growth_rate = -k * (z + f_b * vb) / (adotoa * delta)
    primordial = cosmology.A_s * (k / cosmology.k_p) ** (cosmology.n_s - 1.0)
    pk = 2.0 * jnp.pi**2 / k**3 * primordial * delta**2
    distance = conformal_ages(cosmology)[0] - jnp.interp(A_BINS, a, TAU)
    return pk.T, growth_rate.T, distance, adotoa[0]


_, _, distance_ref, aH_ref = linear_theory(fiducial)


def observed_power(pk, growth_rate, q_perp, q_par, bias):
    """One bin's ``P_obs(k_ref, mu_ref)`` in Mpc^3, ``(len(K_REF), len(MU))``."""

    stretch = jnp.sqrt(1.0 + MU**2 * ((q_perp / q_par) ** 2 - 1.0))
    k = K_REF[:, None] / q_perp * stretch  # the true wave number and angle
    mu = MU * q_perp / q_par / stretch
    pk = jnp.exp(jnp.interp(jnp.log(k), np.log(K), jnp.log(pk)))
    growth_rate = jnp.interp(jnp.log(k), np.log(K), growth_rate)
    return (bias + growth_rate * mu**2) ** 2 * pk / (q_perp**2 * q_par)


def log_power(theta):
    """``ln P_obs`` of every bin, ``(bin, len(K_REF), len(MU))``."""

    cosmology = fiducial.replace(
        **dict(zip(COSMOLOGICAL, theta)), A_s=1.0e-10 * jnp.exp(theta[4])
    )
    pk, growth_rate, distance, aH = linear_theory(cosmology)
    power = jax.vmap(observed_power)(
        pk, growth_rate, distance / distance_ref, aH_ref / aH, theta[-len(BIASES) :]
    )
    return jnp.log(power)


power = jnp.exp(log_power(theta))
derivatives = jax.jacfwd(log_power)(theta)  # (bin, k, mu, parameter)

signal = DENSITIES[:, None, None] * power
weight = (signal / (1.0 + signal)) ** 2 * K_REF[:, None] ** 3  # k^2 dk = k^3 dln k
integrand = jnp.einsum("bkm,bkmi,bkmj->bkmij", weight, derivatives, derivatives)
over_mu = jnp.trapezoid(integrand, MU, axis=2)
per_bin = jnp.trapezoid(over_mu, np.log(K_REF), axis=1)  # (bin, parameter, parameter)
fisher = jnp.einsum("b,bij->ij", VOLUMES / (4.0 * jnp.pi**2), per_bin)

marginalised = np.sqrt(np.diag(np.linalg.inv(fisher)))
conditional = 1.0 / np.sqrt(np.diag(fisher))
print(
    f"{'parameter':>10}  {'fiducial':>10}  {'1 sigma':>10}  {'relative':>9}  {'others fixed':>12}"
)
for name, value, sigma, fixed in zip(NAMES, theta, marginalised, conditional):
    print(
        f"{name:>10}  {value:10.4g}  {sigma:10.2e}  {sigma / value:9.2%}  {fixed:12.2e}"
    )
