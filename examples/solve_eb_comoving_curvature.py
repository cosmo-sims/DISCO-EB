"""Keep one quantity of an Einstein-Boltzmann solve: the comoving curvature perturbation.

``solve_eb`` integrates the perturbation hierarchy of a cosmology for every
wave mode in one GPU launch and records, at the requested conformal times,
only the quantities named in ``keep``. The kernel evaluates them from the
dense-output state at each save, so the device stores ``n_k * n_tau *
len(keep)`` numbers instead of the full 50-variable history. Here that is
the comoving curvature perturbation ``R``: unity at early times, conserved
while a mode is outside the horizon and decaying once it enters.

The kept values are differentiable in the cosmological parameters that enter
the solve through the background densities (Omega_m, Omega_b, H0, T_cmb,
N_eff): asking for a derivative runs the forward-sensitivity launch with a
hook that stores each quantity's derivative beside it, so the Jacobian of the
kept values costs one joint launch and still no history.

Run from the repository root with the project's environment::

    python examples/solve_eb_comoving_curvature.py
"""

import jax
import numpy as np

from discoeb import LCDM, conformal_ages, solve_eb

# Planck 2018 flat LCDM with massless neutrinos.
h = 0.6732117
cosmology = LCDM(
    Omegam=(0.02238280 + 0.1201075) / h**2,
    Omegab=0.02238280 / h**2,
    A_s=2.100549e-9,
    n_s=0.9660499,
    H0=100.0 * h,
    Tcmb=2.7255,
    YHe=0.2454006,
    Neff=3.046,
)

k = np.geomspace(1.0e-4, 1.0, 13)  # 1/Mpc
tau = np.geomspace(1.0, 2.0e4, 200)  # Mpc; the last values lie past tau0

values = solve_eb(cosmology, k, tau, keep=("R",))  # (cosmology, name, k, tau)
R = np.asarray(values[0, 0])  # (n_k, n_tau): one cosmology, one name
tau0 = float(conformal_ages(cosmology)[0])

print(f"conformal age tau0 = {tau0:.1f} Mpc")
print(f"kept {values.nbytes / 1e3:.0f} kB on the device for R;")
print(
    f"the full history at these times would hold "
    f"{len(k) * len(tau) * len(LCDM.state_names()) * 8 / 1e6:.1f} MB"
)
print()
columns = [10.0, 100.0, 1000.0, tau0]
print("R(k, tau)" + "".join(f"{f'tau={c:.0f}':>14}" for c in columns))
for ki, kv in enumerate(k):
    row = np.interp(columns, tau, R[ki])
    print(f"k={kv:8.1e}" + "".join(f"{v:14.6f}" for v in row))
print()
print("Outside the horizon (k tau << 1) every mode keeps R = 1 to O((k tau)^2);")
print("a mode's R falls away once k tau ~ 1 and the potentials decay inside it.")

# The response of today's R to the cosmological parameters, by forward mode:
# d ln R / d ln theta at tau0 for every mode, from one joint launch. The
# thermal history is held fixed (see solve_eb).
import jax.numpy as jnp  # noqa: E402

PARAMETERS = ("Omegam", "Omegab", "H0", "Tcmb", "Neff")


def r_today(log_theta):
    moved = cosmology.replace(**dict(zip(PARAMETERS, jnp.exp(log_theta))))
    return jnp.log(solve_eb(moved, k, tau, keep=("R",))[0, 0, :, -1])


log_theta = jnp.log(jnp.asarray([getattr(cosmology, key) for key in PARAMETERS]))
jacobian = np.asarray(jax.jacfwd(r_today)(log_theta))
print()
print("d ln R(tau0) / d ln " + "".join(f"{n:>12}" for n in PARAMETERS))
for ki, kv in enumerate(k):
    print(f"k={kv:8.1e}" + "".join(f"{v:12.4f}" for v in jacobian[ki]))
