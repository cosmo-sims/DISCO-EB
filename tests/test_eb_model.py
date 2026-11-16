"""The Einstein-Boltzmann component model, on the host.

The equations themselves are gated on the GPU against CLASS
(``test_matter_power_spectrum_matches_class``); here the model's assembly is
checked: that its state order follows the model's components, that each
model builds and runs as plain Python, and that the sums the Einstein
constraints read are the sums over the species the model includes.
"""

import numpy as np
import pytest
from conftest import HOST, MODELS, ConstantThermalHistory, state_index

from discoeb.components.species import DarkEnergy, MassiveNeutrinos, Photons
from discoeb.models import LCDM
from discoeb.solver.kernel import N_PARAM, PARAM_COLUMNS, make_initial_states


def build(cosmology_model):
    model = cosmology_model.build(ConstantThermalHistory())
    return model, model.build(PARAM_COLUMNS, **HOST)


def expected_state_names(model) -> list[str]:
    """Metric, CDM, baryons, photons, massless neutrinos, then the dark-energy fluid and the massive-neutrino bins."""

    names = ["etak", "clxc", "clxb", "vb"]
    names += [f"theta_{ell}" for ell in range(model.setting("LMAX_G") + 1)]
    names += [f"e_{i}" for i in range(model.setting("LMAX_POL") - 1)]  # E_2 ...
    names += [f"n_{ell}" for ell in range(model.setting("LMAX_NR") + 1)]
    if model.has(DarkEnergy):
        names += ["clxq", "thetaq"]
    if model.has(MassiveNeutrinos):
        for q in range(model.setting("NQMAX")):
            names += [f"psi_{q}_{ell}" for ell in range(model.setting("LMAX_NU") + 1)]
    return names


@pytest.mark.parametrize("name", list(MODELS))
def test_the_state_order_follows_the_components(name):
    """Each bin's free-streaming tail ``psi_q_3 .. psi_q_LMAX_NU`` stays contiguous."""
    cosmology_model = MODELS[name]
    model, _ = build(cosmology_model)
    assert list(cosmology_model.state_names()) == expected_state_names(cosmology_model)
    assert list(model.state) == expected_state_names(cosmology_model)


@pytest.mark.parametrize("name", list(MODELS))
def test_the_rhs_builds_and_is_finite(name):
    cosmology_model = MODELS[name]
    model, rhs = build(cosmology_model)
    rng = np.random.default_rng(0)
    p = np.zeros(N_PARAM)
    p[:5] = [1.5e-7, 1.0e-7, 1.1e-4, 1.9e-5, 2.0e-7]
    p[PARAM_COLUMNS["k"]] = 0.05
    p[PARAM_COLUMNS["w_DE_0_nominal"]], p[PARAM_COLUMNS["cs2_DE_nominal"]] = -0.9, 1.0
    (
        p[PARAM_COLUMNS["grhor_nu_nominal"]],
        p[PARAM_COLUMNS["n_mnu"]],
        p[PARAM_COLUMNS["amnu_nominal"]],
    ) = 1e-7, 1.0, 300.0
    y = rng.standard_normal(model.nvar)
    dy = np.asarray(rhs(tuple(y), 100.0, tuple(p)))
    assert dy.shape == (model.nvar,) and np.isfinite(dy).all()


def test_the_einstein_sources_sum_over_the_enabled_species():
    model, _ = build(MODELS["flat-lcdm"])
    lines = [ell.strip() for ell in model.source.splitlines()]
    assert (
        "grho = ColdDarkMatter__grho_c(ColdDarkMatter_self, grhoc_t) + Baryons__grho_b(Baryons_self, grhob_t)"
        " + Photons__grho_g(Photons_self, grhog_t) + MasslessNeutrinos__grho_r(MasslessNeutrinos_self, grhor_t) + CosmologicalConstant__grho_de(CosmologicalConstant_self, grhov_t)"
    ) in lines
    assert (
        "dgq = Baryons__dgq_b(Baryons_self, grhob_t, vb) + Photons__dgq_g(Photons_self, grhog_t, qg) + MasslessNeutrinos__dgq_r(MasslessNeutrinos_self, grhor_t, n, rsa, rsa_q)"
        in lines
    )
    assert "rhonu" not in model.source  # the massive-neutrino table is not read

    model, _ = build(MODELS["ragged-blocks"])
    assert (
        "DarkEnergy__grho_de(DarkEnergy_self, grhov_t) + MassiveNeutrinos__grho_nu(MassiveNeutrinos_self, nu_norm, rhonu)"
        in model.source
    )
    assert (
        "DarkEnergy__dgq_de(DarkEnergy_self, grhov_t, w_Q, thetaq, k) + MassiveNeutrinos__dgq_nu(MassiveNeutrinos_self, nu_norm, psi)"
        in model.source
    )
    assert "psi = cuda.local.array((2, 10), float64)" in model.source
    assert (
        "MassiveNeutrinos__d_psi(MassiveNeutrinos_self, k, z, sigma, a, amnu, psi, inv_tau, c_MassiveNeutrinos__d_psi)"
        in model.source
    )
    assert "d_psi[i, j] += c_MassiveNeutrinos__d_psi[i, j]" in model.source


def test_thomson_scattering_writes_the_photon_and_baryon_derivatives():
    model, _ = build(MODELS["flat-lcdm"])
    lines = [ell.strip() for ell in model.source.splitlines()]
    assert (
        "d_vb = Baryons__d_vb(Baryons_self, adotoa, vb, k, cs2, clxb) + ThomsonScattering__d_vb_thomson(ThomsonScattering_self, photbar, opacity, slip)"
        in lines
    )
    assert lines.index(
        "Photons__d_theta(Photons_self, k, z, sigma, theta, inv_tau, rsa, c_Photons__d_theta)"
    ) < lines.index(
        "ThomsonScattering__d_theta_thomson(ThomsonScattering_self, opacity, slip, polter, theta, rsa, c_ThomsonScattering__d_theta_thomson)"
    )
    assert (
        "ThomsonScattering__d_e_thomson(ThomsonScattering_self, opacity, polter, e, rsa, c_ThomsonScattering__d_e_thomson)"
        in lines
    )
    # Neutrinos are collisionless: one writer.
    assert sum("__d_n(" in ell for ell in lines) == 1


def test_a_hierarchy_needs_an_octupole():
    with pytest.raises(ValueError, match="octupole"):
        build(MODELS["flat-lcdm"].with_settings(LMAX_G=2))


def test_models_derive_from_each_other():
    from discoeb.components.species import CosmologicalConstant
    from discoeb.models import LCDM, LCDM_MNU, W0WACDM, Model

    assert W0WACDM.spec.components.index(DarkEnergy) == LCDM.spec.components.index(
        CosmologicalConstant
    )
    # a model among the bases stands for its components
    assert LCDM_MNU.spec.components == (*LCDM.spec.components, MassiveNeutrinos)
    assert issubclass(LCDM_MNU, LCDM) and LCDM_MNU.spec.name == "LCDM+mnu"
    assert LCDM.with_settings(LMAX_G=20).setting("LMAX_G") == 20
    assert LCDM.with_settings(LMAX_G=20) is LCDM.with_settings(LMAX_G=20)
    assert LCDM.with_settings(LMAX_G=20) != LCDM and LCDM.with_settings() is LCDM
    with pytest.raises(ValueError, match="NQMAX"):
        LCDM.with_settings(NQMAX=4)  # LCDM has no massive neutrinos
    # the run-time derivations are the class statements' equivalents
    assert LCDM.add(MassiveNeutrinos, name="LCDM+mnu").spec == LCDM_MNU.spec
    swapped = LCDM.swap(CosmologicalConstant, DarkEnergy, name="w0waCDM")
    assert swapped.spec == W0WACDM.spec and issubclass(swapped, Model)

    class Coarse(LCDM, LMAX_G=10):
        """A class keyword sets a setting."""

    assert (
        Coarse.setting("LMAX_G") == 10
        and Coarse.spec.components == LCDM.spec.components
    )


BASE = dict(
    Omegam=0.31,
    Omegab=0.049,
    A_s=2.1e-9,
    n_s=0.966,
    H0=67.0,
    Tcmb=2.7255,
    YHe=0.245,
    Neff=3.046,
)


def test_a_model_rejects_a_cosmology_it_cannot_represent():
    from discoeb.models import LCDM, LCDM_MNU, W0WACDM

    LCDM(**BASE)
    LCDM_MNU(**BASE, Nmnu=1.0, mnu=0.06)
    W0WACDM(**BASE, w_DE_0=-0.9)
    with pytest.raises(TypeError, match="no parameter 'w_DE_0'.*DarkEnergy"):
        LCDM(**BASE, w_DE_0=-0.9)  # ty: ignore[unknown-argument]
    with pytest.raises(ValueError, match="w != -1"):
        W0WACDM(**BASE)
    with pytest.raises(TypeError, match="no parameter 'Nmnu'.*MassiveNeutrinos"):
        LCDM(**BASE, Nmnu=1.0)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError, match=r"needs the parameters \['H0'\]"):
        LCDM(**{k: v for k, v in BASE.items() if k != "H0"})
    with pytest.raises(TypeError, match="by keyword"):
        LCDM(BASE)  # ty: ignore[too-many-positional-arguments]
    # from_dict takes a dictionary written for another model, as long as the
    # keys this model lacks keep their defaults
    assert LCDM.from_dict({**BASE, "w_DE_0": -1.0, "unknown": 3.0}).H0 == 67.0
    with pytest.raises(ValueError, match="DarkEnergy"):
        LCDM.from_dict({**BASE, "w_DE_0": -0.9})
    # traced, a key the model lacks can't be checked yet: the cosmology holds
    # it for the solve, which raises if it is differentiated
    import jax

    from discoeb.models._model import _batch_of

    held = jax.jit(lambda w: LCDM.from_dict({**BASE, "w_DE_0": w}))(-1.0)
    assert "w_DE_0" not in held.to_dict() and "w_DE_0" in held._held
    assert _batch_of(held).cosmologies[0]["w_DE_0"] == -1.0


def test_a_cosmology_is_an_immutable_pytree_of_its_parameters():
    import inspect

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM

    planck = LCDM(**BASE)
    assert isinstance(planck, LCDM) and planck.H0 == 67.0 and planck.k_p == 0.05
    assert planck.to_dict() == {**LCDM.parameter_keys(), **BASE} | {
        "k_p": 0.05,
        "Omegak": 0.0,
        "tau_reio": 0.0,
    }
    later = planck.replace(H0=70.0)
    assert later.H0 == 70.0 and planck.H0 == 67.0
    with pytest.raises(AttributeError, match="immutable"):
        planck.H0 = 70.0
    # JAX values are kept as given, so they stay differentiable
    traced = planck.replace(Omegam=jnp.asarray(0.31))
    assert isinstance(traced.Omegam, jax.Array)
    leaves, tree = jax.tree_util.tree_flatten(planck)
    assert leaves == list(planck.to_dict().values())
    doubled = jax.tree_util.tree_unflatten(tree, [2.0 * x for x in leaves])
    assert isinstance(doubled, LCDM) and doubled.H0 == 134.0
    # the constructor's signature and docstring list the parameters
    signature = inspect.signature(LCDM)
    assert list(signature.parameters) == list(LCDM.parameter_keys())
    assert signature.parameters["k_p"].default == 0.05
    assert signature.parameters["H0"].default is inspect.Parameter.empty
    assert "Parameters" in (LCDM.__doc__ or "") and "\nH0 : " in (LCDM.__doc__ or "")


def test_a_solve_takes_cosmologies_of_one_model():
    from discoeb.models import LCDM, W0WACDM
    from discoeb.models._model import _batch_of

    planck = LCDM(**BASE)
    batch = _batch_of([planck, planck.replace(H0=70.0)])
    assert batch.model == LCDM.spec and batch.model_class is LCDM and len(batch) == 2
    assert batch.cosmologies[1]["H0"] == 70.0
    assert len(_batch_of(planck)) == 1
    with pytest.raises(TypeError, match="share a model"):
        _batch_of([planck, W0WACDM(**BASE, w_DE_0=-0.9)])
    with pytest.raises(TypeError, match="non-empty"):
        _batch_of([])
    with pytest.raises(TypeError, match=r"LCDM\(H0="):
        _batch_of([BASE])


def test_the_truncations_are_the_components_settings():
    """The model's lmax settings reach the components, which size their hierarchies."""
    cosmology_model = MODELS["ragged-blocks"]
    model, _ = build(cosmology_model)
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
    model, _ = build(MODELS["ragged-blocks"])
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


def test_make_initial_states_uses_the_model():
    from discoeb.models.cosmology import _as_cosmology

    cosmology = _as_cosmology(
        dict(
            Omegam=0.31,
            Omegab=0.049,
            w_DE_0=-1.0,
            w_DE_a=0.0,
            cs2_DE=1.0,
            Omegak=0.0,
            A_s=2.1e-9,
            n_s=0.966,
            H0=67.0,
            Tcmb=2.7255,
            YHe=0.245,
            Neff=3.046,
            Nmnu=0.0,
            mnu=0.0,
            k_p=0.05,
        )
    )
    from discoeb.solver.kernel import joint_model

    model = joint_model(LCDM)
    y0 = make_initial_states(cosmology, np.array([1e-2, 1e-1]), model, lna_start=-15.0)
    assert y0.shape == (2, 55) and np.isfinite(y0).all()
    np.testing.assert_allclose(y0[:, 0], [-1e-2, -1e-1], rtol=1e-3)  # etak ~ -k
    np.testing.assert_allclose(
        y0[:, 50:], [[-15.0, 1.0, 2.0, 2.7255 * np.exp(15.0), 0.0]] * 2
    )  # the thermal state


def test_the_pressure_sums_over_the_species_and_gives_the_comoving_curvature():
    """``gpres`` is left out of the right-hand side and read by ``R`` alone."""

    for name in ("flat-lcdm", "ragged-blocks"):
        model, _ = build(MODELS[name])
        assert "gpres" not in model.source
    model = MODELS["ragged-blocks"].build(ConstantThermalHistory())
    fn = model.build_function(
        ("R", "gpres", "grho", "adotoa"),
        inputs=("y", "tau"),
        params=PARAM_COLUMNS,
        **HOST,
    )
    lines = [line.strip() for line in fn._generated_source.splitlines()]
    assert (
        "gpres = Photons__gpres_g(Photons_self, grhog_t) + MasslessNeutrinos__gpres_r(MasslessNeutrinos_self, grhor_t)"
        " + DarkEnergy__gpres_de(DarkEnergy_self, grhov_t, w_Q) + MassiveNeutrinos__gpres_nu(MassiveNeutrinos_self, nu_norm, pnu)"
    ) in lines

    # Flat LCDM with the fluids at rest: R is the curvature eta alone, and
    # the velocity term is the total momentum over rho + p.
    cosmology_model = MODELS["flat-lcdm"]
    model = cosmology_model.build(ConstantThermalHistory(a=1.0))
    fn = model.build_function(
        ("R", "gpres", "grho", "adotoa"),
        inputs=("y", "tau"),
        params=PARAM_COLUMNS,
        **HOST,
    )
    p = np.zeros(N_PARAM)
    grhog, grhor, grhoc, grhob, grhov = 1.5e-7, 1.0e-7, 1.1e-4, 1.9e-5, 2.0e-7
    p[:5] = [grhog, grhor, grhoc, grhob, grhov]
    p[PARAM_COLUMNS["k"]] = 0.05
    p[PARAM_COLUMNS["w_DE_0_nominal"]] = -1.0
    y = np.zeros(model.nvar)
    y[state_index(cosmology_model)["etak"]] = -0.05
    y[state_index(cosmology_model)["vb"]] = 0.3
    R, gpres, grho, adotoa = fn(y, 100.0, p)
    assert gpres == pytest.approx((grhog + grhor) / 3.0 - grhov)
    assert grho == pytest.approx(grhog + grhor + grhoc + grhob + grhov)
    dgq = grhob * 0.3
    assert R == pytest.approx(1.0 - adotoa * dgq / (0.05 * (grho + gpres)))
