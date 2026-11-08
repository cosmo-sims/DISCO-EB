"""Time P(k) and the CMB TT, TE and EE spectra of 128 cosmologies.

The matter power spectrum part is ``matter_power_spectrum_batch.py``:
``solve_matter_power_spectrum_batch`` on 128 cosmologies at 128 wave
numbers. The CMB part is ``compute_cl_power_spectrum_batch``: one GPU launch
integrates the perturbations of the same 128 cosmologies on 128 wave modes
and records the line-of-sight sources at 1000 conformal times through a save
hook, and the sources are then projected onto the unlensed scalar TT, TE and
EE ``D_l`` at multipoles 2 to 2500. The CMB grids and projection operator are
built once from a fiducial cosmology, as an inference loop would.

Each calculation's first call compiles its kernel from scratch and is timed on
its own: the script points numba's cache at an empty temporary directory, so
modax's kernel cache is neither read nor written. The second call of each uses
a batch of cosmologies not seen before.

Run from the repository root with the project's environment::

    .venv/bin/python examples/cmb_power_spectra_batch.py

Until modax 0.0.8, which has the Block and Hierarchy API, is released, the
script imports modax from a local checkout at ~/git/modax (hard-coded below)
and DISCO-EB from this repository's src/.
"""

import atexit
import os
import shutil
import sys
import tempfile

# Import modax from the local checkout (the released 0.0.7 lacks the
# hierarchy API) and DISCO-EB from this repository, ahead of any installed
# copies.
sys.path[:0] = [
    os.path.expanduser("~/git/modax"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"),
]

# Compile from fresh every run: an empty kernel cache, removed at exit. This
# must be set before numba or modax is imported.
os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="discoeb-kernel-cache-")
atexit.register(shutil.rmtree, os.environ["NUMBA_CACHE_DIR"], ignore_errors=True)

# The perturbation kernel keeps its own device memory beside JAX's pool.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from discoeb.cmb import cmb_grids, cmb_k_grid, compute_cl_power_spectrum_batch
from discoeb.perturbations import solve_matter_power_spectrum_batch


def cosmologies(n, offset):
    """Flat LCDM, omega_b, omega_c and h spread by under a per cent about Planck 2018."""
    out = []
    for phase in np.linspace(-1.0, 1.0, n) + offset:
        omega_b = 0.02238280 * (1.0 + 0.006 * phase)
        omega_c = 0.1201075 * (1.0 - 0.008 * phase)
        h = 0.6732117 * (1.0 + 0.004 * np.sin(np.pi * phase))
        out.append(
            {
                "Omegam": (omega_b + omega_c) / h**2,
                "Omegab": omega_b / h**2,
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
        )
    return out


k = np.geomspace(1.0e-4, 1.0, 128)  # 1/Mpc
first, fresh = cosmologies(128, offset=0.0), cosmologies(128, offset=0.001)


def timed(f, *args, **kwargs):
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(*args, **kwargs))
    return out, time.perf_counter() - t0


# Matter power spectrum.
_, pk_compile = timed(solve_matter_power_spectrum_batch, k, first)
pk, pk_run = timed(solve_matter_power_spectrum_batch, k, fresh)

# CMB: grids and projection operator from the fiducial, then the spectra.
fiducial = cosmologies(3, offset=0.0)[1]  # the centre of the spread
t0 = time.perf_counter()
grids = cmb_grids(fiducial, k_values=cmb_k_grid(fiducial, n=128), n_save=1000)
grid_time = time.perf_counter() - t0
_, cmb_compile = timed(compute_cl_power_spectrum_batch, first, grids=grids)
(ells, _, dl), cmb_run = timed(compute_cl_power_spectrum_batch, fresh, grids=grids)

n_cosmo, n_k = pk.shape
print(f"P(k), {n_cosmo} cosmologies x {n_k} modes:")
print(f"  compile and first call  {pk_compile:6.1f} s")
print(f"  fresh batch             {pk_run:6.3f} s")
print(
    f"TT, TE, EE, {dl.tt.shape[0]} cosmologies x {len(grids.k_values)} modes, "
    f"{len(ells)} multipoles to l = {ells[-1]}:"
)
print(f"  grids and operator      {grid_time:6.1f} s")
print(f"  compile and first call  {cmb_compile:6.1f} s")
print(f"  fresh batch             {cmb_run:6.3f} s")
