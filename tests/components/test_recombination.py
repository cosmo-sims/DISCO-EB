"""Recombination and reionization (``recombination``), read from the joint kernel, against CLASS."""

import numpy as np
import pytest
from helpers import (
    BATCH_SIZE_CASES,
    REFERENCE_COSMOLOGY,
    _class_base,
    _perturbed_cosmologies,
    cosmologies_of,
)

# --- The recombination history of the perturbation kernel against CLASS, and the time of its solve.


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

    from discoeb import LCDM, TAU_START, conformal_ages, solve_eb

    tau0 = np.asarray(conformal_ages(cosmologies))
    tau = np.geomspace(TAU_START, float(tau0.max()), n_tau)
    values = solve_eb(
        cosmologies_of(LCDM, cosmologies), THERMAL_K, tau, keep=THERMAL_KEEP
    )
    kept = np.asarray(values)[:, :, 0, :]  # one wave mode
    return tau, tau0, {name: kept[:, i] for i, name in enumerate(THERMAL_KEEP)}


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


@pytest.mark.gpu
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


# --- Reionization in the thermal history.


TAU_REIO = 0.0544


LATE = 0.3
"""Conformal times past this fraction of ``tau0`` (``z < ~30``) hold the reionization step;
the recombination tail adds ~1e-4 to their optical depth."""


def _class_z_reio(tau_reio):
    """CLASS's reionization redshift for CAMB's step at ``tau_reio``."""

    Class = pytest.importorskip("classy").Class
    c = Class()
    c.set(
        {
            **_class_base(REFERENCE_COSMOLOGY),
            "recombination": "recfast",
            "reio_parametrization": "reio_camb",
            "tau_reio": tau_reio,
        }
    )
    try:
        c.compute()
        return c.get_current_derived_parameters(["z_reio"])["z_reio"]
    finally:
        c.struct_cleanup()
        c.empty()


def _late_history(tau_reio):
    """``(a, visibility, late optical depth)`` after :data:`LATE` ``tau0``, from the kernel."""

    tau, tau0, history = kernel_thermal_history(
        [{**REFERENCE_COSMOLOGY, "tau_reio": tau_reio}]
    )
    late = tau > LATE * tau0[0]
    kappa = history["kappa"][0]
    visibility = history["opacity"][0] * np.exp(-(kappa[-1] - kappa))
    return history["a"][0][late], visibility[late], float(kappa[-1] - kappa[late][0])


@pytest.mark.gpu
def test_reionization_optical_depth():
    """The late optical depth is tau_reio, and the bump sits where CLASS puts z_reio."""

    a, g, tau_late = _late_history(TAU_REIO)
    assert abs(tau_late - TAU_REIO) < 1.0e-3, tau_late
    z_bump = 1.0 / a[np.argmax(g)] - 1.0
    # the visibility peaks a little below z_reio (the step is half way there); 1.5 in z
    z_reio_class = _class_z_reio(TAU_REIO)
    assert abs(z_bump - z_reio_class) < 1.5, (z_bump, z_reio_class)


@pytest.mark.gpu
def test_no_reionization_by_default():
    """tau_reio = 0 (the default) adds no optical depth after recombination."""

    _, _, tau_late = _late_history(0.0)
    assert tau_late < 5.0e-4
