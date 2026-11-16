"""Set up differentiable initial conditions for a DISCO-DJ N-body simulation.

An N-body simulation starts from one Gaussian realisation of the linear
matter field. Inflation sets a single random field, the comoving curvature
perturbation R, and every species follows from it through a transfer
function. DISCO-EB's solutions are per unit initial R, so at the start time

    delta_m(k, tau_ini) = T_m(k, tau_ini) R(k),
    <|R(k)|^2> = 2 pi^2 / k^3 A_s (k / k_p)^(n_s - 1),

and the particles move off their grid by the Zel'dovich displacement
Psi(k) = i k delta_m(k) / k^2.

Every step is JAX, so the initial conditions are differentiable in the
cosmological parameters at a fixed seed. Below, the derivative of every
particle's displacement with respect to Omega_m costs one forward-sensitivity
launch, and DISCO-DJ can carry it on through the simulation.

The start is a fixed conformal time, so the derivative is exact. Its scale
factor depends on the cosmology and comes out beside the displacements.
Lengths are in Mpc.

Run from the repository root with the project's environment::

    uv run python examples/setup_n_body_ics.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from discoeb import LCDM, solve_eb

N_GRID = 64  # particles per side
BOX = 500.0  # side of the box, Mpc
TAU_INI = 2000.0  # conformal start time, Mpc: z of about 50

# Wave numbers of the grid, and a solve grid that covers them.
k_axis = 2.0 * np.pi * np.fft.fftfreq(N_GRID, d=BOX / N_GRID)
k_half = 2.0 * np.pi * np.fft.rfftfreq(N_GRID, d=BOX / N_GRID)
k_vec = jnp.stack(jnp.meshgrid(k_axis, k_axis, k_half, indexing="ij"))
k_mag = jnp.sqrt(jnp.sum(k_vec**2, axis=0)).at[0, 0, 0].set(1.0)  # no k = 0 mode
K_SOLVE = np.geomspace(np.pi / BOX, 2.0 * np.sqrt(3.0) * np.pi * N_GRID / BOX, 128)


def initial_conditions(cosmology, white_noise):
    """Zel'dovich displacements ``(3, N, N, N)`` in Mpc and the start's scale factor."""

    values = solve_eb(cosmology, K_SOLVE, [TAU_INI], keep=("clxc", "clxb", "a"))
    clxc, clxb, a = values[0, :, :, 0]  # each (n_k,), per unit initial R
    Omegam, Omegab = cosmology.Omegam, cosmology.Omegab
    transfer = ((Omegam - Omegab) * clxc + Omegab * clxb) / Omegam
    transfer = jnp.interp(jnp.log(k_mag), jnp.log(K_SOLVE), transfer)

    # R on the grid: white noise coloured by the primordial spectrum.
    power_R = (
        2.0
        * jnp.pi**2
        / k_mag**3
        * cosmology.A_s
        * (k_mag / cosmology.k_p) ** (cosmology.n_s - 1.0)
    )
    R = jnp.fft.rfftn(white_noise) * jnp.sqrt(power_R / (BOX / N_GRID) ** 3)

    delta = (transfer * R).at[0, 0, 0].set(0.0)
    psi = jnp.fft.irfftn(1j * k_vec * delta / k_mag**2, s=white_noise.shape)
    return psi, a[0]


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
white_noise = jax.random.normal(jax.random.key(42), (N_GRID,) * 3)

displacement, a_ini = initial_conditions(planck, white_noise)

# The same particles' response to Omega_m, at the same seed.
d_displacement, d_a_ini = jax.jacfwd(
    lambda Omegam: initial_conditions(planck.replace(Omegam=Omegam), white_noise)
)(jnp.asarray(planck.Omegam))
