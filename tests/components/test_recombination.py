"""Recombination and reionization (``recombination``), read from the joint kernel, against CLASS."""

import numpy as np
import pytest
from helpers import (
    BATCH_SIZE_CASES,
    LCDM_REIO,
    REFERENCE_COSMOLOGY,
    _class_base,
    _perturbed_cosmologies,
    cosmologies_of,
)

# --- The recombination history of the perturbation kernel against CLASS, and the time of its solve.


THERMAL_KEEP = ("tau", "xe", "Tmat", "cs2", "opacity", "kappa")


THERMAL_K = np.array([1.0e-3])
"""One wave mode per cosmology: the recombination block does not read the hierarchy."""


N_A = 2000


Z_MIN, Z_MAX = 10.0, 5000.0


XE_GATE = 0.01


TM_GATE = 0.002


CS2_GATE = 0.01


SWITCH_WINDOW = (800.0, 900.0)


CS2_SWITCH_GATE = 0.2


def kernel_thermal_history(cosmologies, model=None, keep=THERMAL_KEEP, n_a: int = N_A):
    """``(a, history)`` of the kernel's recombination block on a logarithmic grid of the scale factor.

    ``a`` runs from :data:`discoeb.A_START` to 1, today for every cosmology.
    ``history[name]`` is ``(n_cosmologies, n_a)`` for each name of ``keep``.
    """

    from discoeb import A_START, LCDM, solve_eb

    batch = cosmologies_of(LCDM if model is None else model, cosmologies)
    a = np.geomspace(A_START, 1.0, n_a)
    values = solve_eb(batch, THERMAL_K, a, keep=keep)
    kept = np.asarray(values)[:, :, 0, :]  # one wave mode
    return a, {name: kept[:, i] for i, name in enumerate(keep)}


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
    a, history = benchmark.pedantic(
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
        z = 1.0 / a - 1.0
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


# --- The conformal time, a state of the block.


def test_the_conformal_time_starts_from_the_radiation_eras_closed_form():
    """``tau`` at the start is the integral of ``da / (a^2 H)`` from zero, with no integration."""

    from helpers import on_the_host

    from discoeb import A_START, LCDM, LCDM_MNU
    from discoeb.components.recombination import _tau_initial

    for model, more in ((LCDM, {}), (LCDM_MNU, {"Nmnu": 1.0, "mnu": 0.3})):
        name = model.__name__
        cosmology = {**REFERENCE_COSMOLOGY, **more}
        rate, matter = on_the_host(model, cosmology, "adotoa", "grho_matter", a=A_START)
        closed = _tau_initial(float(rate), float(matter))
        # in a, where the integrand 1 / (a^2 H) is nearly constant; below
        # the grid the universe is radiation alone and tau = 1 / adotoa
        grid = np.linspace(1.0e-3 * A_START, A_START, 4000)
        rates = np.asarray(on_the_host(model, cosmology, "adotoa", a=grid))
        integral = np.trapezoid(1.0 / (grid * rates), grid) + 1.0 / rates[0]
        assert closed == pytest.approx(integral, rel=1e-6), name
        assert 0.05 < closed < 0.2


# --- Reionization in the thermal history.


Z_RE = 7.67
"""Planck 2018's midpoint of CAMB's step, for ``tau_reio`` near 0.054."""


LATE = 1.0 / 31.0
"""Scale factors past this one (``z < 30``) hold the reionization step; the
recombination tail adds ~1e-4 to their optical depth."""


def _class_tau_reio(z_re):
    """CLASS's optical depth of CAMB's step at ``z_re``."""

    Class = pytest.importorskip("classy").Class
    c = Class()
    c.set(
        {
            **_class_base(REFERENCE_COSMOLOGY),
            "recombination": "recfast",
            "reio_parametrization": "reio_camb",
            "z_reio": z_re,
        }
    )
    try:
        c.compute()
        return c.get_current_derived_parameters(["tau_reio"])["tau_reio"]
    finally:
        c.struct_cleanup()
        c.empty()


def _late_history(z_re):
    """``(a, visibility, late optical depth, tau_reio)`` after :data:`LATE`, from the kernel.

    Of LCDM with :class:`~discoeb.components.recombination.ReionizationDepth`,
    whose state ``kappa_reio`` is ``tau_reio`` today.
    """

    a, history = kernel_thermal_history(
        [{**REFERENCE_COSMOLOGY, "z_re": z_re}],
        model=LCDM_REIO,
        keep=(*THERMAL_KEEP, "kappa_reio"),
    )
    late = a > LATE
    kappa = history["kappa"][0]
    visibility = history["opacity"][0] * np.exp(-(kappa[-1] - kappa))
    return (
        a[late],
        visibility[late],
        float(kappa[-1] - kappa[late][0]),
        float(history["kappa_reio"][0][-1]),
    )


@pytest.mark.gpu
def test_reionization_optical_depth():
    """``kappa_reio`` today is CLASS's tau_reio at the same z_re, and the bump sits at z_re."""

    a, g, tau_late, tau_reio = _late_history(Z_RE)
    tau_class = _class_tau_reio(Z_RE)
    assert abs(tau_reio - tau_class) < 1.0e-3, (tau_reio, tau_class)
    # the late optical depth is reionization's plus the recombination tail
    assert 0.0 < tau_late - tau_reio < 5.0e-4, (tau_late, tau_reio)
    z_bump = 1.0 / a[np.argmax(g)] - 1.0
    # the visibility peaks a little below z_re (the step is half way there)
    assert abs(z_bump - Z_RE) < 1.5, z_bump


@pytest.mark.gpu
def test_no_reionization_by_default():
    """z_re = 0 (the default) adds no optical depth after recombination."""

    _, _, tau_late, tau_reio = _late_history(0.0)
    assert tau_late < 5.0e-4 and tau_reio == 0.0
