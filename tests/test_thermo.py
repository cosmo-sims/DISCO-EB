"""The recombination history of the perturbation kernel against CLASS, and the time of its solve.

RECFAST runs as the recombination block of the joint kernel
(:mod:`discoeb.recombination`); :func:`kernel_thermal_history` records its
free-electron fraction, matter temperature, baryon sound speed, opacity and
optical depth through :func:`discoeb.solve_eb` at one wave mode per
cosmology. The check runs the same cosmologies as
``test_matter_power_spectrum_matches_class``: one, and a batch of 128 nearby
flat LCDM models, and compares ``x_e``, ``T_m`` and ``c_b^2`` with CLASS run
with its own recfast implementation and no reionization.

The gates are set from the measured agreement, with margin. Over every 8th
cosmology of the batch, the electron fraction differs from CLASS by at most
0.36 percent (hydrogen leaves Saha equilibrium on redshift as in CLASS 3), the
temperature by 0.04 percent and ``c_b^2`` by 0.4 percent, except in the
redshift window where RECFAST switches off its tight Compton-coupling
approximation for ``T_m`` (``timeTh < H_frac * timeH``): there ``dT_m/dz``
jumps momentarily, which CLASS's full treatment does not reproduce, and
``c_b^2`` differs by up to 9.7 percent around ``z ~ 850``.

The benchmark times one warmed :func:`kernel_thermal_history` of the batch.
"""

import numpy as np
import pytest
from test_perturbations import BATCH_SIZE_CASES, _class_base, _perturbed_cosmologies

THERMAL_KEEP = ("a", "xe", "Tmat", "cs2", "opacity", "kappa")
THERMAL_K = np.array([1.0e-3])
"""One wave mode per cosmology: the recombination block does not read the hierarchy."""
N_TAU = 2000

Z_MIN, Z_MAX = 10.0, 5000.0
XE_GATE = 0.01
TM_GATE = 0.002
CS2_GATE = 0.01
SWITCH_WINDOW = (800.0, 900.0)
CS2_SWITCH_GATE = 0.2


def kernel_thermal_history(cosmologies, n_tau: int = N_TAU):
    """``(tau, tau0, history)`` of the kernel's recombination block on a ``log tau`` grid.

    ``tau`` runs from :data:`discoeb.TAU_START` to the batch's largest
    conformal age; ``history[name]`` is ``(n_cosmologies, n_tau)`` for each
    of :data:`THERMAL_KEEP`, a cosmology's values holding past its own ``tau0``.
    """

    from discoeb import TAU_START, conformal_ages, solve_eb

    tau0 = np.asarray(conformal_ages(cosmologies))
    tau = np.geomspace(TAU_START, float(tau0.max()), n_tau)
    solution = solve_eb(cosmologies, THERMAL_K, tau, keep=THERMAL_KEEP)
    return (
        tau,
        tau0,
        {name: np.asarray(solution[name])[:, 0, :] for name in THERMAL_KEEP},
    )


def _class_thermodynamics(param):
    """Return ``(z, x_e, T_b, c_b^2)`` from CLASS with recfast and no reionization, ascending in z."""

    Class = pytest.importorskip("classy").Class
    cosmo = Class()
    cosmo.set(
        {
            **_class_base(param),
            "recombination": "recfast",
            "reio_parametrization": "reio_none",
        }
    )
    try:
        cosmo.compute()
        thermo = cosmo.get_thermodynamics()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    order = np.argsort(thermo["z"])
    return tuple(
        np.asarray(thermo[key])[order] for key in ("z", "x_e", "Tb [K]", "c_b^2")
    )


@pytest.mark.parametrize("n_cosmologies", BATCH_SIZE_CASES)
def test_recfast_matches_class(n_cosmologies, benchmark):
    """Check the kernel's recombination history against CLASS and time one warmed batch solve."""

    cosmologies = _perturbed_cosmologies(n_cosmologies)
    _, _, history = benchmark.pedantic(
        kernel_thermal_history,
        args=(cosmologies,),
        rounds=1,
        warmup_rounds=1,
        iterations=1,
    )

    def worst(ours, theirs, z, mask):
        rel = np.abs(ours[mask] / theirs[mask] - 1.0)
        i = np.argmax(rel)
        return float(rel[i]), float(z[mask][i])

    for i, param in enumerate(cosmologies):
        z = 1.0 / history["a"][i] - 1.0
        inside = (z >= Z_MIN) & (z <= Z_MAX)
        switch = (z > SWITCH_WINDOW[0]) & (z < SWITCH_WINDOW[1])
        z_class, xe_class, tb_class, cb2_class = _class_thermodynamics(param)
        xe_ref = np.interp(z, z_class, xe_class)
        tm_ref = np.interp(z, z_class, tb_class)
        cs2_ref = np.interp(z, z_class, cb2_class)
        assert np.isfinite(history["xe"][i]).all()

        err, at = worst(history["xe"][i], xe_ref, z, inside)
        assert err < XE_GATE, f"cosmology {i}: x_e off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["Tmat"][i], tm_ref, z, inside)
        assert err < TM_GATE, f"cosmology {i}: T_m off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["cs2"][i], cs2_ref, z, inside & ~switch)
        assert err < CS2_GATE, f"cosmology {i}: c_b^2 off by {err:.3g} at z={at:.0f}"
        err, at = worst(history["cs2"][i], cs2_ref, z, inside & switch)
        assert err < CS2_SWITCH_GATE, (
            f"cosmology {i}: c_b^2 off by {err:.3g} at z={at:.0f} in the tight-coupling switch window"
        )
