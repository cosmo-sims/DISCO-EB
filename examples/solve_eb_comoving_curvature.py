"""Keep one quantity of an Einstein-Boltzmann solve: the comoving curvature perturbation.

``solve_eb`` integrates the perturbation hierarchy of a cosmology for every
wave mode in one GPU launch and records, at the requested conformal times,
only the quantities named in ``keep``. The kernel evaluates them from the
dense-output state at each save, so the device stores ``n_k * n_tau *
len(keep)`` numbers instead of the full 50-variable history. Here that is
the comoving curvature perturbation ``R``: unity at early times, conserved
while a mode is outside the horizon and decaying once it enters.

Run from the repository root with the project's environment::

    python examples/solve_eb_comoving_curvature.py
"""

import os

# The perturbation kernel keeps its own device memory beside JAX's pool.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from discoeb.perturbations import solve_eb

# Planck 2018 flat LCDM with massless neutrinos.
h = 0.6732117
cosmology = {
    "Omegam": (0.02238280 + 0.1201075) / h**2,
    "Omegab": 0.02238280 / h**2,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omegak": 0.0,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 100.0 * h,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
    "Nmnu": 0.0,
    "mnu": 0.0,
    "k_p": 0.05,
}

k = np.geomspace(1.0e-4, 1.0, 13)  # 1/Mpc
tau = np.geomspace(1.0, 2.0e4, 200)  # Mpc; the last values lie past tau0

solution = solve_eb(cosmology, k, tau, keep=("R",))
R = np.asarray(solution["R"][0])  # (n_k, n_tau); one cosmology was passed
tau0 = float(solution.tau0[0])

print(f"conformal age tau0 = {tau0:.1f} Mpc")
print(f"kept {solution.values.nbytes / 1e3:.0f} kB on the device for {solution.names};")
print(
    f"the full history at these times would hold "
    f"{len(k) * len(tau) * solution.layout.nvar * 8 / 1e6:.1f} MB"
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
