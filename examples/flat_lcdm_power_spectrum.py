"""Compute the linear matter power spectrum of a flat LCDM cosmology.

An entry point to DISCO-EB: pick a model from ``discoeb.models``, apply it
to a cosmology given as a parameter dictionary and solve. The script prints ``P(k)`` at
redshift zero and, with ``--output``, writes ``k`` and ``P(k)`` to a text
file. The first run compiles the GPU kernel (about 35 s); modax caches it on
disk, so later runs start in seconds.

Run from the repository root with the project's environment::

    .venv/bin/python examples/flat_lcdm_power_spectrum.py
    .venv/bin/python examples/flat_lcdm_power_spectrum.py --h 0.7 --output pk.txt
"""

import argparse
import os

# The perturbation kernel keeps its own device memory beside JAX's pool.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

# Before DISCO-EB builds anything: the solver works in double precision.
jax.config.update("jax_enable_x64", True)

import numpy as np

from discoeb import LCDM, solve_matter_power_spectrum


def flat_lcdm(omega_b: float, omega_c: float, h: float, A_s: float, n_s: float):
    """A flat LCDM parameter dictionary with three massless neutrino flavours."""

    return {
        "Omegam": (omega_b + omega_c) / h**2,
        "Omegab": omega_b / h**2,
        "H0": 100.0 * h,
        "A_s": A_s,
        "n_s": n_s,
        "k_p": 0.05,  # pivot scale of A_s, 1/Mpc
        "Tcmb": 2.7255,
        "YHe": 0.2454006,
        "Neff": 3.046,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Planck 2018 best fit by default.
    parser.add_argument("--omega-b", type=float, default=0.02238280)
    parser.add_argument("--omega-c", type=float, default=0.1201075)
    parser.add_argument("--h", type=float, default=0.6732117)
    parser.add_argument("--A-s", type=float, default=2.100549e-9)
    parser.add_argument("--n-s", type=float, default=0.9660499)
    parser.add_argument("--k-min", type=float, default=1.0e-4, help="1/Mpc")
    parser.add_argument("--k-max", type=float, default=1.0, help="1/Mpc")
    parser.add_argument("--n-k", type=int, default=64)
    parser.add_argument("--output", help="write k [1/Mpc] and P(k) [Mpc^3] here")
    args = parser.parse_args(argv)

    cosmology = flat_lcdm(args.omega_b, args.omega_c, args.h, args.A_s, args.n_s)
    k = np.geomspace(args.k_min, args.k_max, args.n_k)

    # One GPU launch over every wave mode; a list of cosmologies batches them.
    (pk,) = solve_matter_power_spectrum(k, LCDM([cosmology]))

    print(f"{'k [1/Mpc]':>12}  {'P(k) [Mpc^3]':>14}")
    for kv, pv in zip(k[:: max(1, len(k) // 16)], pk[:: max(1, len(k) // 16)]):
        print(f"{kv:12.4e}  {pv:14.6e}")
    print(f"P(k) peaks at k = {k[np.argmax(pk)]:.4g} 1/Mpc")

    if args.output:
        np.savetxt(
            args.output,
            np.column_stack([k, pk]),
            header="k [1/Mpc]  P(k) [Mpc^3], linear, z = 0",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
