"""Reionization in the thermal history.

The thermodynamics tables take an optical depth ``tau_reio`` and add CAMB's
tanh reionization step to the RECFAST ionization fraction, with the
reionization redshift set by bisection on the optical depth.  Checked here:
the integrated Thomson opacity of the low-redshift step reproduces the
requested optical depth, the reionization redshift agrees with CLASS's for the
same ``tau_reio`` (both use the same parametrization), and ``tau_reio = 0``
leaves the history untouched. DISCO-CMB checks the spectra with reionization
against CLASS.
"""

import numpy as np
import pytest
from test_perturbations import REFERENCE_COSMOLOGY

TAU_REIO = 0.0544


def _cuda_available() -> bool:
    try:
        from numba_cuda_mlir import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _class(param, extra):
    """A computed CLASS instance for ``param`` with CAMB-style reionization at its ``tau_reio``."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    c = Class()
    c.set(
        {
            "h": h,
            "omega_b": param["Omegab"] * h**2,
            "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
            "A_s": param["A_s"],
            "n_s": param["n_s"],
            "k_pivot": param["k_p"],
            "YHe": param["YHe"],
            "T_cmb": param["Tcmb"],
            "N_ur": param["Neff"],
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "recombination": "recfast",
            "reio_parametrization": "reio_camb",
            "tau_reio": param["tau_reio"],
            **extra,
        }
    )
    c.compute()
    return c


def _tables(tau_reio):
    """The thermal history and its visibility ``g = kappa' exp(-kappa)``, ``kappa`` from ``tau0``."""

    from discoeb.perturbations import _as_cosmology, host_thermal_history

    cosmology = _as_cosmology({**REFERENCE_COSMOLOGY, "tau_reio": tau_reio})
    tau_grid, values, tau0 = host_thermal_history(cosmology)
    opacity = values[1]
    steps = 0.5 * (opacity[1:] + opacity[:-1]) * np.diff(tau_grid)
    kappa = np.concatenate([np.cumsum(steps[::-1])[::-1], [0.0]])
    emk = np.exp(-kappa)
    return tau_grid, values, float(tau0), opacity * emk, emk


def test_reionization_optical_depth():
    """exp(-kappa) today is exp(-tau_reio), and the bump sits where CLASS puts z_reio."""

    tau_grid, values, tau0, g, emk = _tables(TAU_REIO)
    late = (
        tau_grid > 0.3 * tau0
    )  # z < ~30: the reionization step; the recombination tail adds ~1e-4
    tau_late = np.trapezoid(values[1][late], tau_grid[late])
    assert abs(tau_late - TAU_REIO) < 1.0e-3, tau_late

    # the visibility has a secondary bump after recombination, at CLASS's z_reio
    c = _class({**REFERENCE_COSMOLOGY, "tau_reio": TAU_REIO}, {})
    try:
        z_reio_class = c.get_current_derived_parameters(["z_reio"])["z_reio"]
    finally:
        c.struct_cleanup()
        c.empty()
    a_bump = values[0][late][np.argmax(g[late])]
    z_bump = 1.0 / a_bump - 1.0
    # the visibility peaks a little below z_reio (the step is half way there); 1.5 in z
    assert abs(z_bump - z_reio_class) < 1.5, (z_bump, z_reio_class)


def test_no_reionization_by_default():
    """tau_reio = 0 (the default) leaves exp(-kappa) at unity today."""

    tau_grid, values, tau0, _, _ = _tables(0.0)
    late = tau_grid > 0.3 * tau0
    assert np.trapezoid(values[1][late], tau_grid[late]) < 5.0e-4
