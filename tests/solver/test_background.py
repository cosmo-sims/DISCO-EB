"""The expansion history (``_background``): densities, the budget closure and the background states."""

import math

import numpy as np
import pytest
from helpers import (
    COSMOLOGY,
    LAM,
    LCDM_T,
    LCDM_X,
    LCDM_XY,
    OMEGA_X,
    QCDM,
    QCDM_COSMOLOGY,
    _qcdm_densities,
    _split,
)

from discoeb.models import LCDM

# --- One component listed several times, each copy under a name of its own.


def test_the_background_closes_with_both_densities():
    from discoeb.models._cosmology import _columns
    from discoeb.solver._background import density_parameters

    def densities(model, cosmology):
        columns = _columns([cosmology], held=False, keys=model.parameter_keys())
        return {
            k: float(v[0]) for k, v in density_parameters(columns, model=model).items()
        }

    lcdm = densities(LCDM, COSMOLOGY)
    both = densities(
        LCDM_XY,
        {
            **COSMOLOGY,
            "Omegam": COSMOLOGY["Omegam"] - 0.05,
            "x_Omega": 0.03,
            "y_Omega": 0.02,
        },
    )
    assert both["x_grho0"] == pytest.approx(0.03 * both["grhom"], rel=1e-15)
    assert both["grhoc"] + both["x_grho0"] + both["y_grho0"] == pytest.approx(
        lcdm["grhoc"], rel=1e-14
    )
    assert both["grhov"] == pytest.approx(lcdm["grhov"], rel=1e-12)


# --- A species with a parameter of its own, added without touching the package.


def test_the_background_closes_the_budget_with_the_new_density():
    from discoeb.models._cosmology import _columns
    from discoeb.solver._background import density_parameters

    def densities(model, cosmology):
        columns = _columns([cosmology], held=False, keys=model.parameter_keys())
        return {
            k: float(v[0]) for k, v in density_parameters(columns, model=model).items()
        }

    lcdm = densities(LCDM, COSMOLOGY)
    split = densities(LCDM_X, _split(COSMOLOGY))
    assert split["grhox"] == pytest.approx(OMEGA_X * split["grhom"], rel=1e-15)
    assert split["grhoc"] + split["grhox"] == pytest.approx(lcdm["grhoc"], rel=1e-14)
    assert split["grhov"] == pytest.approx(lcdm["grhov"], rel=1e-12)
    assert "grhox" not in lcdm


# --- A background state added from outside: the equations split the blocks themselves.


def test_the_host_integrates_the_states_of_the_background():
    """``t = int a dtau``, by Runge-Kutta steps beside the conformal time."""

    import jax.numpy as jnp

    from discoeb.models._cosmology import _columns
    from discoeb.solver._background import (
        _background_solution,
        background_endpoints,
        density_parameters,
    )

    columns = _columns([COSMOLOGY], held=False, keys=LCDM_T.parameter_keys())
    densities = {k: v[0] for k, v in density_parameters(columns, model=LCDM_T).items()}
    a, tau, states = _background_solution(densities, LCDM_T)
    a_lcdm, tau_lcdm, none = _background_solution(densities, LCDM)
    assert none == {}
    # the conformal time of LCDM, whose trapezoid rule is the less exact (2e-6)
    np.testing.assert_allclose(tau, tau_lcdm, rtol=5e-6)
    t = np.concatenate(([0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(tau_lcdm))))
    np.testing.assert_allclose(states["t"][1:], t[1:], rtol=1e-4)
    tau0, _, start = background_endpoints(columns, LCDM_T)
    assert set(start) == {"t"}
    assert float(start["t"][0]) == pytest.approx(
        float(jnp.interp(0.1, tau, states["t"])), rel=1e-12
    )
    assert float(tau0[0]) == pytest.approx(float(tau_lcdm[-1]), rel=5e-6)


# --- A scalar field from outside the package: background states, and a budget closed by shooting.


def test_a_flat_potential_closes_the_budget_as_a_cosmological_constant():
    lcdm = _qcdm_densities(LCDM, QCDM_COSMOLOGY)
    flat = _qcdm_densities(QCDM, QCDM_COSMOLOGY)
    assert float(flat["V0"]) == pytest.approx(float(lcdm["grhov"]), rel=1e-9)


@pytest.mark.slow
def test_shooting_closes_the_budget_of_a_rolling_field():
    """``V0`` is the root of ``H(a = 1) = H0`` through the field's integration."""

    import jax
    import jax.numpy as jnp

    from discoeb.solver._background import _background_solution, background_model

    d = _qcdm_densities(QCDM, {**QCDM_COSMOLOGY, "lam": LAM})
    a, tau, states = _background_solution(d, QCDM)
    today = {name: values[-1] for name, values in states.items()}
    adotoa, grho, gpres = background_model(QCDM).evaluate(
        "adotoa", "grho", "gpres", math=jnp, a=1.0, **today, **d
    )
    assert float(3.0 * adotoa**2 / d["grhom"]) == pytest.approx(1.0, abs=1e-10)
    # the field has rolled: more potential than LCDM's constant to start with,
    # and an equation of state above -1 today
    assert float(d["V0"]) > float(_qcdm_densities(LCDM, QCDM_COSMOLOGY)["grhov"])
    kinetic = 0.5 * float(today["phidot"]) ** 2
    potential = float(d["V0"]) * math.exp(-LAM * float(today["phi"]))
    w0 = (kinetic - potential) / (kinetic + potential)
    assert -0.95 < w0 < -0.7
    assert float(today["phi"]) > 0.0

    # and the closure is differentiable: more matter leaves less for the field
    def V0_of(Omegam):
        return _qcdm_densities(QCDM, {**QCDM_COSMOLOGY, "lam": LAM, "Omegam": Omegam})[
            "V0"
        ]

    Omegam = QCDM_COSMOLOGY["Omegam"]
    slope = float(jax.jacfwd(V0_of)(jnp.asarray(Omegam)))
    h = 1e-4
    central = float(V0_of(Omegam + h) - V0_of(Omegam - h)) / (2 * h)
    assert slope < 0.0
    assert slope == pytest.approx(central, rel=1e-5)
