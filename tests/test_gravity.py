"""The gravity interface: what the species read, and a theory replacing the one that provides it."""

import math

import numpy as np
import pytest
from conftest import HOST, ConstantThermalHistory
from jax.typing import ArrayLike

from discoeb.components._component import (
    Component,
    Equations,
    Gravity,
    Parameter,
    State,
)
from discoeb.components.gravity import GRAVITY_INTERFACE, Background, Curvature, Metric
from discoeb.components.species import (
    ColdDarkMatter,
    NoRadiationStreaming,
    RadiationStreaming,
)
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
    from discoeb.models._cosmology import _columns
    from discoeb.solver._background import density_parameters

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

    from discoeb.solver._background import background_model

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

    from discoeb.solver._background import background_model

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

    mu: ArrayLike = Parameter(1.0, column=True)

    def adotoa(self, grho, Kcurv, mu):
        return math.sqrt(mu * grho * (1.0 / 3.0) - Kcurv)


def test_another_friedmann_equation_replaces_the_background():
    """The species are untouched, and the budget closes with the new rate."""

    import jax.numpy as jnp

    from discoeb.solver._background import background_model

    scaled = LCDM.swap(Background, ScaledGravity, name="scaled gravity")
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
    from discoeb.solver._kernel import parameter_layout

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

    model = Equations([Tank(), Refill(), Gauge()])
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
        Equations([Loop()]).build({}, **HOST)


def test_the_metric_rate_is_a_quantity_of_the_model():
    from discoeb.solver._kernel import N_PARAM, PARAM_COLUMNS

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


def test_the_gauge_needs_its_species_at_rest():
    assert ColdDarkMatter.fixes_gauge and not Metric.fixes_gauge
    with pytest.raises(ValueError, match="fixes the synchronous gauge"):
        LCDM.remove(ColdDarkMatter, name="no CDM")
    # a copy under another name fixes it as well
    LCDM.swap(ColdDarkMatter, ColdDarkMatter.named("c"), name="renamed CDM")


def test_the_streaming_approximation_is_a_component_a_model_may_drop():
    """No species names it, and no species is named by it."""

    from discoeb.solver._kernel import N_PARAM, PARAM_COLUMNS

    full = LCDM.swap(RadiationStreaming, NoRadiationStreaming, name="full")
    assert full.state_names() == LCDM.state_names()
    assert "RSA" not in {n for cls in full.spec.components for n in cls.setting_names()}
    with pytest.raises(NameError, match="rsa"):
        LCDM.remove(RadiationStreaming, name="nothing").build(
            ConstantThermalHistory()
        ).build(PARAM_COLUMNS, **HOST)

    def rhs_of(cosmology_model):
        model = cosmology_model.build(ConstantThermalHistory(opacity=1.0e-6))
        return model, model.build(PARAM_COLUMNS, **HOST)

    p = np.zeros(N_PARAM)
    for name, value in dict(
        grhog=1.5e-7, grhornomass=1.0e-7, grhoc=0.8e-4, grhob=1.9e-5, grhov=2.0e-7
    ).items():
        p[PARAM_COLUMNS[f"{name}_nominal"]] = value
    p[PARAM_COLUMNS["k"]] = 0.5
    model, rhs = rhs_of(full)
    y = np.random.default_rng(2).standard_normal(model.nvar)
    tau = 5000.0  # deep inside the horizon and decoupled: the approximation is on
    without = np.asarray(rhs(tuple(y), tau, tuple(p)))
    _, off = rhs_of(LCDM.with_settings(RSA=0))
    _, on = rhs_of(LCDM)
    np.testing.assert_array_equal(without, np.asarray(off(tuple(y), tau, tuple(p))))
    assert not np.array_equal(without, np.asarray(on(tuple(y), tau, tuple(p))))

    # each streaming species adds its own density to the sum
    streaming = LCDM.build(ConstantThermalHistory()).build_function(
        ("dgrho_streaming", "tau"), inputs=("y", "tau"), params=PARAM_COLUMNS, **HOST
    )
    assert (
        "dgrho_streaming = Photons__dgrho_streaming_g(Photons_self, rsa, rsa_delta, grhog_t)"
        " + MasslessNeutrinos__dgrho_streaming_r(MasslessNeutrinos_self, rsa, rsa_delta, grhor_t)"
    ) in streaming._generated_source
