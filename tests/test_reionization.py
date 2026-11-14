"""Reionization in the thermal history.

A cosmology's ``tau_reio`` adds CAMB's tanh reionization step to the RECFAST
ionization fraction of the kernel's recombination block, with the
reionization redshift set by bisection on the optical depth
(:func:`discoeb.perturbations.background_endpoints`). Checked here, on the
kernel's own opacity and optical depth (:func:`test_thermo.kernel_thermal_history`):
the optical depth of the low-redshift step reproduces the requested
``tau_reio``, the visibility's reionization bump sits at CLASS's reionization
redshift for the same ``tau_reio`` (both use the same parametrization), and
``tau_reio = 0`` adds no late optical depth. DISCO-CMB checks the spectra with
reionization against CLASS.
"""

import numpy as np
import pytest
from test_perturbations import REFERENCE_COSMOLOGY, _class_base
from test_thermo import kernel_thermal_history

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


def test_reionization_optical_depth():
    """The late optical depth is tau_reio, and the bump sits where CLASS puts z_reio."""

    a, g, tau_late = _late_history(TAU_REIO)
    assert abs(tau_late - TAU_REIO) < 1.0e-3, tau_late
    z_bump = 1.0 / a[np.argmax(g)] - 1.0
    # the visibility peaks a little below z_reio (the step is half way there); 1.5 in z
    z_reio_class = _class_z_reio(TAU_REIO)
    assert abs(z_bump - z_reio_class) < 1.5, (z_bump, z_reio_class)


def test_reionization_redshift_derivative():
    """``z_re`` is differentiable in ``tau_reio``, and fixed where the bracket clamps it.

    Inside the bisection's bracket ``[1, 40]`` the derivative is the implicit
    one and matches a central difference. A ``tau_reio`` too small to reach
    ``z_re = 1`` leaves ``z_re`` at the bound, so its derivative is zero, as
    the clamped value's is.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.perturbations import (
        _as_cosmology,
        _stack_background_params,
        background_endpoints,
    )

    def z_re(tau_reio):
        cosmology = _as_cosmology({**REFERENCE_COSMOLOGY, "tau_reio": 0.0}, jnp.asarray)
        stacked = _stack_background_params((cosmology,))
        return background_endpoints({**stacked, "tau_reio": tau_reio[None]})[2][0]

    step = 1.0e-4
    derivative = float(jax.grad(z_re)(jnp.asarray(TAU_REIO)))
    finite = (
        float(z_re(jnp.asarray(TAU_REIO + step)))
        - float(z_re(jnp.asarray(TAU_REIO - step)))
    ) / (2.0 * step)
    assert derivative > 0.0
    assert abs(derivative - finite) < 1.0e-3 * abs(finite), (derivative, finite)

    assert float(z_re(jnp.asarray(1.0e-3))) == pytest.approx(1.0, abs=1.0e-6)
    assert float(jax.grad(z_re)(jnp.asarray(1.0e-3))) == 0.0


def test_no_reionization_by_default():
    """tau_reio = 0 (the default) adds no optical depth after recombination."""

    _, _, tau_late = _late_history(0.0)
    assert tau_late < 5.0e-4
