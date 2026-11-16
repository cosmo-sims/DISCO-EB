"""The expansion history (``_background``): densities and the background states."""

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
    _split,
    cosmologies_of,
    qcdm_parameters,
)

from discoeb.components.gravity import GRHO_PER_OMEGA
from discoeb.models import LCDM


def densities(model, cosmology):
    """The model's parameters, derived ones included, of a dictionary in ``Omegam`` and ``H0``."""

    from discoeb.solver._background import density_parameters

    native = cosmologies_of(model, cosmology).to_dict()
    return density_parameters(native, xp=np, model=model.spec)


H2 = (COSMOLOGY["H0"] / 100.0) ** 2


# --- One component listed several times, each copy under a name of its own.


def test_the_budget_closes_with_both_densities():
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
    assert both["x_grho0"] == pytest.approx(GRHO_PER_OMEGA * 0.03 * H2, rel=1e-15)
    assert both["grhoc"] + both["x_grho0"] + both["y_grho0"] == pytest.approx(
        lcdm["grhoc"], rel=1e-14
    )
    assert both["grhov"] == pytest.approx(lcdm["grhov"], rel=1e-12)


# --- A species with a parameter of its own, added without touching the package.


def test_the_budget_closes_with_the_new_density():
    lcdm = densities(LCDM, COSMOLOGY)
    split = densities(LCDM_X, _split(COSMOLOGY))
    assert split["grhox"] == pytest.approx(GRHO_PER_OMEGA * OMEGA_X * H2, rel=1e-15)
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

    native = cosmologies_of(LCDM_T, COSMOLOGY).to_dict()
    columns = _columns([native], held=False, keys=LCDM_T.parameter_keys())
    d = {k: v[0] for k, v in density_parameters(columns, model=LCDM_T).items()}
    a, tau, states = _background_solution(d, LCDM_T)
    a_lcdm, tau_lcdm, none = _background_solution(d, LCDM)
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


# --- A scalar field from outside the package: background states, and no budget to close.


def test_a_scalar_field_has_no_hubble_constant_before_the_solve():
    """Its potential is a parameter, and the expansion rate today reads the field."""

    flat = QCDM.from_dict(qcdm_parameters())
    lcdm = cosmologies_of(LCDM, QCDM_COSMOLOGY)
    assert flat.V0 == pytest.approx(GRHO_PER_OMEGA * lcdm.omega_de, rel=1e-15)
    assert lcdm.H0 == pytest.approx(QCDM_COSMOLOGY["H0"], rel=1e-12)
    with pytest.raises(TypeError, match="needs a solve"):
        flat.H0
    with pytest.raises(TypeError, match="no dark-energy density"):
        QCDM.from_density_parameters(**QCDM_COSMOLOGY)


@pytest.mark.slow
def test_a_rolling_field_expands_more_slowly_today():
    """At LCDM's potential height a rolling field has lost density by today: ``H(a = 1) < H0``."""

    import jax.numpy as jnp

    from discoeb.solver._background import (
        _background_solution,
        background_model,
        density_parameters,
    )

    d = density_parameters(qcdm_parameters(lam=LAM), model=QCDM.spec)
    a, tau, states = _background_solution(d, QCDM)
    today = {name: values[-1] for name, values in states.items()}
    adotoa = background_model(QCDM.spec).evaluate(
        "adotoa", math=jnp, a=1.0, **today, **d
    )
    h0 = float(adotoa) * math.sqrt(3.0 / (GRHO_PER_OMEGA * 1.0e-4))
    assert 0.9 * QCDM_COSMOLOGY["H0"] < h0 < QCDM_COSMOLOGY["H0"]
    kinetic = 0.5 * float(today["phidot"]) ** 2
    potential = float(d["V0"]) * math.exp(-LAM * float(today["phi"]))
    w0 = (kinetic - potential) / (kinetic + potential)
    assert -0.95 < w0 < -0.7
    assert float(today["phi"]) > 0.0
