"""The species (``species``): their hierarchies, truncations, initial conditions and the named ``Fluid``."""

import numpy as np
import pytest
from helpers import (
    HOST,
    LCDM_XY,
    MODELS,
    ConstantThermalHistory,
    build_rhs,
)

from discoeb.components.species import ColdDarkMatter, MassiveNeutrinos, Photons
from discoeb.models import LCDM

# --- One component listed several times, each copy under a name of its own.


def test_a_cold_pressureless_copy_moves_as_the_dark_matter():
    """``w = cs2 = 0`` at rest: ``x_delta' = -k z``, the dark matter's equation."""

    from discoeb.solver._kernel import parameter_layout

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
        p[columns[f"{name}"]] = value
    p[columns["x_grho0"]] = 0.3e-4
    p[columns["y_grho0"]] = 0.1e-4
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
    """A second cold dark matter, with a density ``warm_omega_c`` of its own."""

    second = ColdDarkMatter.named("warm")
    parameters = dict(second.declared_parameters())
    assert parameters["warm_grhoc"].reads == ("warm_omega_c",)
    assert LCDM.add(second, name="two").state_names()[-1] == "warm_clxc"


# --- The Einstein-Boltzmann component model, on the host.


def test_a_hierarchy_needs_an_octupole():
    with pytest.raises(ValueError, match="octupole"):
        build_rhs(MODELS["flat-lcdm"].with_settings(LMAX_G=2))


def test_the_truncations_are_the_components_settings():
    """The model's lmax settings reach the components, which size their hierarchies."""
    cosmology_model = MODELS["ragged-blocks"]
    model, _ = build_rhs(cosmology_model)
    photons = next(c for c in model.components if isinstance(c, Photons))
    assert (photons.LMAX_G, photons.LMAX_POL) == (11, 13)
    assert (model.states["theta"].shape, model.states["e"].shape) == ((12,), (12,))
    neutrinos = next(c for c in model.components if isinstance(c, MassiveNeutrinos))
    assert (neutrinos.NQMAX, neutrinos.LMAX_NU) == (2, 9)
    assert model.states["psi"].shape == (2, 10) and len(neutrinos.weight) == 2
    # The defaults are the DISCO_LMAX_* environment settings, 15 unless set.
    assert MODELS["flat-lcdm"].setting("LMAX_G") == Photons.LMAX_G


def test_the_initial_state_and_sparsity_come_from_the_components():
    """The adiabatic mode: etak ~ -k, matter at 3/4 of the photon density, zero above the octupole."""
    model, _ = build_rhs(MODELS["ragged-blocks"])
    densities = (1.5e-7, 1.0e-7, 1.1e-4, 1.9e-5, 2.0e-7)
    k = np.array([1e-3, 1e-1])
    y0 = model.initial_state(
        k,
        tau=0.1,
        grhog=densities[0],
        grhornomass=densities[1],
        grhoc=densities[2],
        grhob=densities[3],
        grhov=densities[4],
        w_DE_0=-0.9,
        w_DE_a=0.1,
    )
    assert y0.shape == (2, model.nvar) and np.isfinite(y0).all()
    np.testing.assert_allclose(y0[:, model.index["etak"]], -k, rtol=1e-3)
    np.testing.assert_allclose(
        y0[:, model.index["clxc"]], 0.75 * y0[:, model.index["theta_0"]]
    )
    np.testing.assert_allclose(
        y0[:, model.index["clxq"]], (1.0 - 0.9) * y0[:, model.index["clxc"]]
    )
    assert (y0[:, model.index["theta_4"] : model.index["e_0"]] == 0.0).all()
    assert (y0[:, model.index["n_4"] : model.index["n_17"] + 1] == 0.0).all()
    neutrinos = next(c for c in model.components if isinstance(c, MassiveNeutrinos))
    np.testing.assert_allclose(
        y0[:, model.index["psi_1_0"]],
        -0.25 * y0[:, model.index["n_0"]] * neutrinos.dlnf0[1],
    )
    # The sparsity pattern is traced through the same components: the metric
    # reads the dark-energy momentum through dgq, a free-streaming neutrino
    # multipole reads only its neighbours.
    pattern = {(int(r), int(c)) for r, c in model.sparsity()}
    i = model.index
    assert (i["etak"], i["thetaq"]) in pattern and (i["n_5"], i["n_4"]) in pattern
    assert (i["n_5"], i["etak"]) not in pattern and (i["n_5"], i["n_5"]) not in pattern
