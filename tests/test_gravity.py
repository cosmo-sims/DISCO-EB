"""The gravity interface: what the species read, and a theory replacing the one that provides it."""

import math

import numpy as np
import pytest
from conftest import HOST, ConstantThermalHistory

from discoeb.component import Component, Gravity, Model, Parameter, State
from discoeb.gravity import GRAVITY_INTERFACE, Background, Curvature, Metric
from discoeb.models import LCDM, W0WACDM_MNU

COSMOLOGY = {
    "Omegam": 0.3,
    "Omegab": 0.05,
    "A_s": 2.1e-9,
    "n_s": 0.96,
    "H0": 67.0,
    "Tcmb": 2.7255,
    "YHe": 0.245,
    "Neff": 3.046,
}


def _densities(model, cosmology):
    from discoeb.background import density_parameters
    from discoeb.cosmology import _columns

    columns = _columns([cosmology], held=False, keys=model.parameter_keys())
    return {k: v[0] for k, v in density_parameters(columns, model=model).items()}


def test_a_model_needs_the_whole_interface():
    assert GRAVITY_INTERFACE == ("adotoa", "z", "sigma")
    assert issubclass(Background, Gravity) and issubclass(Metric, Gravity)
    with pytest.raises(ValueError, match=r"provides \['z', 'sigma'\]"):
        LCDM.remove(Metric, name="no metric")
    with pytest.raises(ValueError, match=r"provides \['adotoa'\]"):
        LCDM.remove(Background, name="no Friedmann equation")


def test_curvature_is_geometry_and_not_a_fluid():
    """It reaches the expansion rate through ``Kcurv`` and none of the sums."""

    import jax.numpy as jnp

    from discoeb.background import background_model

    assert issubclass(Curvature, Gravity)
    curved = {**COSMOLOGY, "Omegak": 0.05}
    d = _densities(LCDM, curved)
    adotoa, grho = background_model(LCDM).evaluate(
        "adotoa", "grho", math=jnp, a=1.0, **d
    )
    # the fluids fall short of the critical density by the curvature's share
    assert float(grho / d["grhom"]) == pytest.approx(0.95, rel=1e-12)
    assert float(3.0 * adotoa**2 / d["grhom"]) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("a", [1.0e-4, 1.0e-2, 0.5, 1.0])
def test_the_acceleration_equation_is_the_derivative_of_the_friedmann_equation(a):
    """``adotoa_prime = -(grho + 3 gpres) / 6`` for every species, curved or not."""

    import jax
    import jax.numpy as jnp

    from discoeb.background import background_model

    cosmology = {
        **COSMOLOGY,
        "Omegak": -0.03,
        "w_DE_0": -0.9,
        "w_DE_a": 0.2,
        "Nmnu": 1.0,
        "mnu": 0.3,
    }
    d = _densities(W0WACDM_MNU, cosmology)
    background = background_model(W0WACDM_MNU)

    def adotoa(lna):
        return background.evaluate("adotoa", math=jnp, a=jnp.exp(lna), **d)

    lna = jnp.log(a)
    rate, slope = jax.jvp(adotoa, (lna,), (jnp.ones(()),))
    prime = background.evaluate("adotoa_prime", math=jnp, a=a, **d)
    # d/dtau = (a'/a) d/dln a
    assert float(prime) == pytest.approx(float(rate * slope), rel=1e-9)


class ScaledGravity(Background):
    """A Friedmann equation with a rescaled Newton constant: ``H^2 = mu 8 pi G rho / 3 - K``."""

    mu = Parameter(1.0, column=True)

    def adotoa(self, grho, Kcurv, mu):
        return math.sqrt(mu * grho * (1.0 / 3.0) - Kcurv)


def test_another_friedmann_equation_replaces_the_background():
    """The species are untouched, and the budget closes with the new rate."""

    import jax.numpy as jnp

    from discoeb.background import background_model

    scaled = LCDM.replace(Background, ScaledGravity, name="scaled gravity")
    assert scaled.state_names() == LCDM.state_names()
    assert "mu" in scaled.parameter_keys() and "mu" not in LCDM.parameter_keys()
    plain = _densities(LCDM, COSMOLOGY)
    d = _densities(scaled, {**COSMOLOGY, "mu": 1.1})
    adotoa = background_model(scaled).evaluate("adotoa", math=jnp, a=1.0, **d)
    # the model still expands at H0 today, with less dark energy to do it
    assert float(3.0 * adotoa**2 / d["grhom"]) == pytest.approx(1.0, rel=1e-12)
    assert float(d["grhov"]) < float(plain["grhov"])
    total = 1.1 * (d["grhov"] + d["grhoc"] + d["grhob"] + d["grhog"] + d["grhornomass"])
    assert float(total / d["grhom"]) == pytest.approx(1.0, rel=1e-12)
    # and the right-hand side builds with it
    from discoeb.kernel import parameter_layout

    model = scaled.build(ConstantThermalHistory())
    model.build(parameter_layout(scaled).columns, **HOST)
    assert "ScaledGravity__adotoa(ScaledGravity_self, grho, Kcurv, mu)" in model.source


def test_a_rate_is_a_quantity_a_provider_may_read():
    """``d_<state>`` of a scalar state: the sum of what writes it."""

    class Tank(Component):
        level = State()

        def d_level(self, level):
            return -0.5 * level

    class Refill(Component):
        def d_level(self):
            return 2.0

    class Gauge(Component):
        trend = State()

        def falling(self, d_level):
            return -d_level

        def d_trend(self, falling):
            return falling

    model = Model([Tank(), Refill(), Gauge()])
    rhs = model.build({}, **HOST)
    assert rhs((3.0, 0.0), 0.0, ()) == (0.5, -0.5)
    # computed once, as a quantity, and returned as the state's rate
    assert model.source.count("d_level = ") == 1
    assert model.evaluate("falling", level=3.0) == -0.5
    fn = model.build_function(("d_level", "falling"), inputs=("y",), **HOST)
    assert fn((3.0, 0.0)) == (0.5, -0.5)
    # the trend reads the level through its rate
    assert model.sparsity().tolist() == [[0, 0], [1, 0]]

    class Loop(Component):
        x = State()

        def d_x(self, d_x):
            return d_x

    with pytest.raises(ValueError, match="circular"):
        Model([Loop()]).build({}, **HOST)


def test_the_metric_rate_is_a_quantity_of_the_model():
    from discoeb.kernel import N_PARAM, PARAM_COLUMNS

    model = LCDM.build(ConstantThermalHistory())
    rhs = model.build(PARAM_COLUMNS, **HOST)
    fn = model.build_function(
        ("d_etak", "adotoa_prime"), inputs=("y", "tau"), params=PARAM_COLUMNS, **HOST
    )
    p = np.zeros(N_PARAM)
    for name, value in dict(
        grhog=1.5e-7, grhornomass=1.0e-7, grhoc=0.8e-4, grhob=1.9e-5, grhov=2.0e-7
    ).items():
        p[PARAM_COLUMNS[f"{name}_nominal"]] = value
    p[PARAM_COLUMNS["k"]] = 0.05
    y = np.random.default_rng(1).standard_normal(model.nvar)
    d_etak, adotoa_prime = fn(tuple(y), 100.0, tuple(p))
    assert d_etak == rhs(tuple(y), 100.0, tuple(p))[model.index["etak"]]
    assert adotoa_prime < 0.0  # matter and radiation decelerate
