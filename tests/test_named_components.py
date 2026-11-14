"""One component listed several times, each copy under a name of its own.

``Fluid`` is a generic perfect fluid. ``Fluid.named("x")`` is a copy whose
states, parameters and quantities carry the prefix ``x_``, so a model may
hold any number of them without a class written for each.
"""

import numpy as np
import pytest
from conftest import HOST, ConstantThermalHistory, _cuda_available

from discoeb.component import Component, Model, State
from discoeb.models import LCDM
from discoeb.species import ColdDarkMatter, Fluid

X, Y = Fluid.named("x"), Fluid.named("y")
LCDM_XY = LCDM.add(X, Y, name="LCDM+x+y")

COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
}
OMEGA_X = 0.08


def test_a_named_copy_prefixes_what_the_component_declares():
    assert Fluid.named("x") is X and X is not Y
    assert issubclass(X, Fluid) and X.__name__ == "Fluid_x"
    assert [n for n, _ in X.declared_states()] == ["x_delta", "x_theta"]
    assert [n for n, _ in X.declared_parameters()] == [
        "x_Omega",
        "x_w",
        "x_cs2",
        "x_grho0",
    ]
    # the derived density reads the copy's own Omega and the model's grhom
    assert dict(X.declared_parameters())["x_grho0"].reads == ("grhom", "x_Omega")
    with pytest.raises(TypeError, match="already carries a name"):
        X.named("z")
    with pytest.raises(ValueError, match="not a name"):
        Fluid.named("not a name")


def test_two_copies_share_a_model():
    from discoeb.kernel import parameter_layout

    keys = LCDM_XY.parameter_keys()
    assert {"x_Omega", "x_w", "x_cs2", "y_Omega", "y_w", "y_cs2"} <= set(keys)
    assert "Omega" not in keys and "x_Omega" not in LCDM.parameter_keys()
    names = LCDM_XY.state_names()
    assert names[-4:] == ("x_delta", "x_theta", "y_delta", "y_theta")
    layout = parameter_layout(LCDM_XY)
    assert layout.directed[-2:] == ("x_grho0", "y_grho0")
    assert {"x_w", "y_cs2", "x_grho0_nominal", "dir_y_grho0"} <= set(layout.columns)
    # unnamed, the fluid's theta is the photons'
    with pytest.raises(ValueError, match="theta"):
        LCDM.add(Fluid, name="clash").build()


def test_each_copy_checks_its_own_keys():
    LCDM_XY({**COSMOLOGY, "x_Omega": 0.01, "y_w": 1.0 / 3.0})
    with pytest.raises(ValueError, match="w != -1"):
        LCDM_XY({**COSMOLOGY, "y_w": -1.0})
    with pytest.raises(ValueError, match="use a model with Fluid_x"):
        LCDM({**COSMOLOGY, "x_Omega": 0.01})


def test_the_background_closes_with_both_densities():
    from discoeb.background import density_parameters
    from discoeb.cosmology import _columns

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


def test_a_cold_pressureless_copy_moves_as_the_dark_matter():
    """``w = cs2 = 0`` at rest: ``x_delta' = -k z``, the dark matter's equation."""

    from discoeb.kernel import parameter_layout

    columns = parameter_layout(LCDM_XY).columns
    model = LCDM_XY.build(ConstantThermalHistory())
    rhs = model.build(columns, **HOST)
    assert "x_grho_t = Fluid_x__grho_t(Fluid_x_self, x_grho0, a, x_w)" in [
        line.strip() for line in model.source.splitlines()
    ]
    p = np.zeros(len(columns))
    for name, value in dict(
        grhog=1.5e-7, grhornomass=1.0e-7, grhoc=0.8e-4, grhob=1.9e-5, grhov=2.0e-7
    ).items():
        p[columns[f"{name}_nominal"]] = value
    p[columns["x_grho0_nominal"]] = 0.3e-4
    p[columns["y_grho0_nominal"]] = 0.1e-4
    p[columns["y_w"]], p[columns["y_cs2"]] = 1.0 / 3.0, 1.0 / 3.0
    p[columns["k"]] = 0.05
    y = np.random.default_rng(0).standard_normal(model.nvar)
    y[model.index["x_theta"]] = 0.0
    dy = np.asarray(rhs(tuple(y), 100.0, tuple(p)))
    assert np.isfinite(dy).all()
    assert dy[model.index["x_delta"]] == dy[model.index["clxc"]]
    assert dy[model.index["x_theta"]] == 0.0
    # the radiation copy has pressure: its velocity feels its own contrast
    assert dy[model.index["y_theta"]] != 0.0
    pattern = {(int(r), int(c)) for r, c in model.sparsity()}
    assert (model.index["y_theta"], model.index["y_delta"]) in pattern
    assert (model.index["y_theta"], model.index["x_delta"]) not in pattern
    assert (model.index["etak"], model.index["x_delta"]) in pattern


def test_a_built_in_species_takes_a_name_too():
    """A second cold dark matter, with its own ``Omegam`` and the model's ``Omegab``."""

    second = ColdDarkMatter.named("warm")
    parameters = dict(second.declared_parameters())
    assert parameters["warm_grhoc"].reads == ("grhom", "warm_Omegam", "Omegab")
    assert LCDM.add(second, name="two").state_names()[-1] == "warm_clxc"


def test_names_resolve_in_initial_conditions_and_sums():
    class Tank(Component):
        level = State(initial=lambda start: 2.0 * start)

        def start(self):
            return 1.5

        def outflow(self, level):
            return 0.1 * level

        def total(self, level):
            return level

        def d_level(self, outflow):
            return -outflow

    model = Model([Tank.named("a")(), Tank.named("b")()], sums=("total",))
    assert model.state == ["a_level", "b_level"]
    np.testing.assert_allclose(model.initial_state(np.array([1.0])), [[3.0, 3.0]])
    rhs = model.build({}, **HOST)
    assert rhs((1.0, 2.0), 0.0, ()) == (-0.1, -0.2)
    assert model.evaluate("total", a_level=1.0, b_level=2.0) == 3.0
    assert model.evaluate("d_b_level", a_level=1.0, b_level=2.0) == -0.2


@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")
def test_splitting_the_dark_matter_into_a_named_fluid_leaves_the_solve_unchanged():
    from discoeb import solve_eb

    model = LCDM.add(X, name="LCDM+x")
    k = np.geomspace(1.0e-3, 0.3, 8)
    tau = np.geomspace(1.0, 2.0e4, 16)
    plain = solve_eb(LCDM(COSMOLOGY), k, tau, keep=("clxc", "R"))
    split = solve_eb(
        model(
            {**COSMOLOGY, "Omegam": COSMOLOGY["Omegam"] - OMEGA_X, "x_Omega": OMEGA_X}
        ),
        k,
        tau,
        keep=("clxc", "R", "x_delta"),
    )
    np.testing.assert_allclose(split.tau0, plain.tau0, rtol=1e-12)
    np.testing.assert_allclose(split["x_delta"], split["clxc"], rtol=1e-5)
    for name in ("clxc", "R"):
        a, b = np.asarray(split[name]), np.asarray(plain[name])
        assert np.abs(a - b).max() < 1e-3 * np.abs(b).max(), name
