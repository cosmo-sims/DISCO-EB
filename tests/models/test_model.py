"""The model base classes (``_model``): model classes, specs, cosmologies as instances, the sparsity pattern."""

import numpy as np
import pytest
from helpers import (
    COSMOLOGY,
    HOST,
    LCDM_X,
    MODELS,
    N_PARAM,
    PARAM_COLUMNS,
    ConstantThermalHistory,
    _split,
    build_rhs,
    cosmologies_of,
    dense_core_pattern,
    diagonal,
    entries,
    nvar,
    state_index,
    structure,
    ty_errors,
    ty_executable,
)

from discoeb.components.species import DarkEnergy, MassiveNeutrinos
from discoeb.models import LCDM

# --- The Einstein-Boltzmann component model, on the host.


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
    model, _ = build_rhs(cosmology_model)
    assert list(cosmology_model.state_names()) == expected_state_names(cosmology_model)
    assert list(model.state) == expected_state_names(cosmology_model)


@pytest.mark.parametrize("name", list(MODELS))
def test_the_rhs_builds_and_is_finite(name):
    cosmology_model = MODELS[name]
    model, rhs = build_rhs(cosmology_model)
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


def test_each_component_marks_its_differentiable_keys():
    """A key is differentiable where its declaration says so, and only there."""

    from helpers import LCDM_REIO, LCDM_X

    from discoeb.components._component import ParameterDeclaration
    from discoeb.models import LCDM_MNU, W0WACDM, W0WACDM_MNU

    lcdm = {"omega_b", "omega_c", "omega_de", "Tcmb", "Neff", "omega_k"}
    assert set(LCDM.spec.differentiable_parameters()) == lcdm
    assert set(LCDM_MNU.spec.differentiable_parameters()) == lcdm | {"mnu"}
    dark_energy = lcdm | {"w_DE_0", "w_DE_a"}
    assert set(W0WACDM.spec.differentiable_parameters()) == dark_energy
    assert set(W0WACDM_MNU.spec.differentiable_parameters()) == dark_energy | {"mnu"}
    # z_re only where ReionizationDepth resolves its derivative
    assert set(LCDM_REIO.spec.differentiable_parameters()) == lcdm | {"z_re"}
    # a component from outside marks none unless it says so
    assert set(LCDM_X.spec.differentiable_parameters()) == lcdm
    assert "Differentiable through solve_eb." in LCDM.__doc__.split("omega_c :")[1]
    assert "Differentiable" not in LCDM.__doc__.split("YHe :")[1].split("\n")[1]
    with pytest.raises(TypeError, match="only a key"):
        ParameterDeclaration(value=lambda omega_c: omega_c, differentiable=True)


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


BASE_COSMOLOGY = dict(
    omega_b=0.022,
    omega_c=0.117,
    omega_de=0.31,
    A_s=2.1e-9,
    n_s=0.966,
    Tcmb=2.7255,
    YHe=0.245,
    Neff=3.046,
)


def test_a_model_rejects_a_cosmology_it_cannot_represent():
    from discoeb.models import LCDM, LCDM_MNU, W0WACDM

    LCDM(**BASE_COSMOLOGY)
    LCDM_MNU(**BASE_COSMOLOGY, Nmnu=1.0, mnu=0.06)
    W0WACDM(**BASE_COSMOLOGY, w_DE_0=-0.9)
    with pytest.raises(TypeError, match="no parameter 'w_DE_0'.*DarkEnergy"):
        LCDM(**BASE_COSMOLOGY, w_DE_0=-0.9)  # ty: ignore[unknown-argument]
    with pytest.raises(ValueError, match="w != -1"):
        W0WACDM(**BASE_COSMOLOGY)
    with pytest.raises(TypeError, match="no parameter 'Nmnu'.*MassiveNeutrinos"):
        LCDM(**BASE_COSMOLOGY, Nmnu=1.0)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError, match=r"needs the parameters \['omega_c'\]"):
        LCDM(**{k: v for k, v in BASE_COSMOLOGY.items() if k != "omega_c"})
    with pytest.raises(TypeError, match="by keyword"):
        LCDM(BASE_COSMOLOGY)  # ty: ignore[too-many-positional-arguments]
    # from_dict takes a dictionary written for another model, as long as the
    # keys this model lacks keep their defaults
    assert (
        LCDM.from_dict({**BASE_COSMOLOGY, "w_DE_0": -1.0, "unknown": 3.0}).omega_c
        == 0.117
    )
    with pytest.raises(ValueError, match="DarkEnergy"):
        LCDM.from_dict({**BASE_COSMOLOGY, "w_DE_0": -0.9})
    # traced, a key the model lacks can't be checked yet: the cosmology holds
    # it for the solve, which raises if it is differentiated
    import jax

    from discoeb.models._model import _batch_of

    held = jax.jit(lambda w: LCDM.from_dict({**BASE_COSMOLOGY, "w_DE_0": w}))(-1.0)
    assert "w_DE_0" not in held.to_dict() and "w_DE_0" in held._held
    assert _batch_of(held).cosmologies[0]["w_DE_0"] == -1.0


def test_a_cosmology_is_an_immutable_pytree_of_its_parameters():
    import inspect

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM

    planck = LCDM(**BASE_COSMOLOGY)
    assert isinstance(planck, LCDM) and planck.omega_c == 0.117 and planck.k_p == 0.05
    assert planck.to_dict() == {**LCDM.parameter_keys(), **BASE_COSMOLOGY} | {
        "k_p": 0.05,
        "omega_k": 0.0,
        "z_re": 0.0,
    }
    later = planck.replace(omega_c=0.13)
    assert later.omega_c == 0.13 and planck.omega_c == 0.117
    with pytest.raises(AttributeError, match="immutable"):
        planck.omega_c = 0.13
    # JAX values are kept as given, so they stay differentiable
    traced = planck.replace(omega_c=jnp.asarray(0.117))
    assert isinstance(traced.omega_c, jax.Array)
    leaves, tree = jax.tree_util.tree_flatten(planck)
    assert leaves == list(planck.to_dict().values())
    doubled = jax.tree_util.tree_unflatten(tree, [2.0 * x for x in leaves])
    assert isinstance(doubled, LCDM) and doubled.omega_c == 0.234
    # the constructor's signature and docstring list the parameters
    signature = inspect.signature(LCDM)
    assert list(signature.parameters) == list(LCDM.parameter_keys())
    assert signature.parameters["k_p"].default == 0.05
    assert signature.parameters["omega_c"].default is inspect.Parameter.empty
    assert "Parameters" in (LCDM.__doc__ or "") and "\nomega_c : " in (
        LCDM.__doc__ or ""
    )


def test_density_parameters_give_the_physical_ones_and_the_hubble_constant_back():
    """``from_density_parameters`` closes the budget, and ``H0`` is what the densities sum to."""

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM, LCDM_MNU, W0WACDM

    fractions = dict(Omegam=0.31, Omegab=0.049, H0=67.0)
    rest = {k: v for k, v in BASE_COSMOLOGY.items() if not k.startswith("omega")}
    planck = LCDM.from_density_parameters(**fractions, **rest)
    h2 = 0.67**2
    assert planck.omega_b == pytest.approx(0.049 * h2, rel=1e-15)
    assert planck.omega_c == pytest.approx((0.31 - 0.049) * h2, rel=1e-15)
    assert planck.H0 == pytest.approx(67.0, rel=1e-13)
    # dark energy takes what matter and radiation leave of the critical density
    assert 0.68 * h2 < planck.omega_de < 0.69 * h2
    # curvature, massive neutrinos and a fluid with w != -1 close the same way
    for model, more in (
        (LCDM, {"Omegak": 0.02}),
        (LCDM_MNU, {"Nmnu": 1.0, "mnu": 0.3}),
        (W0WACDM, {"w_DE_0": -0.9, "w_DE_a": 0.1}),
    ):
        other = model.from_density_parameters(**fractions, **rest, **more)
        assert other.H0 == pytest.approx(67.0, rel=1e-13), model.__name__
    assert LCDM.from_density_parameters(
        **fractions, **rest, Omegak=0.02
    ).omega_k == pytest.approx(0.02 * h2, rel=1e-15)
    with pytest.raises(TypeError, match="omega_de or H0"):
        LCDM.from_density_parameters(**fractions, **rest, omega_de=0.3)
    # plain JAX arithmetic: a derivative in H0 goes through the closure
    slope = jax.grad(
        lambda H0: (
            LCDM.from_density_parameters(**{**fractions, "H0": H0}, **rest).omega_de
        )
    )(jnp.asarray(67.0))
    assert float(slope) == pytest.approx(2.0 * planck.omega_de / 67.0, rel=1e-2)


def test_the_budget_closes_with_densities_added_from_outside():
    """A named fluid's and a new species' share comes out of the dark energy's, as fractions of the critical density."""

    from helpers import (
        COSMOLOGY,
        LCDM_X,
        LCDM_XY,
        OMEGA_X,
        _split,
        model_parameters,
    )

    from discoeb.components.gravity import GRHO_PER_OMEGA
    from discoeb.models import LCDM

    h2 = (COSMOLOGY["H0"] / 100.0) ** 2
    lcdm = model_parameters(LCDM, COSMOLOGY)
    both = model_parameters(
        LCDM_XY,
        {
            **COSMOLOGY,
            "Omegam": COSMOLOGY["Omegam"] - 0.05,
            "x_Omega": 0.03,
            "y_Omega": 0.02,
        },
    )
    assert both["x_grho0"] == pytest.approx(GRHO_PER_OMEGA * 0.03 * h2, rel=1e-15)
    assert both["grhoc"] + both["x_grho0"] + both["y_grho0"] == pytest.approx(
        lcdm["grhoc"], rel=1e-14
    )
    assert both["grhov"] == pytest.approx(lcdm["grhov"], rel=1e-12)
    split = model_parameters(LCDM_X, _split(COSMOLOGY))
    assert split["grhox"] == pytest.approx(GRHO_PER_OMEGA * OMEGA_X * h2, rel=1e-15)
    assert split["grhoc"] + split["grhox"] == pytest.approx(lcdm["grhoc"], rel=1e-14)
    assert split["grhov"] == pytest.approx(lcdm["grhov"], rel=1e-12)
    assert "grhox" not in lcdm


def test_a_scalar_field_has_no_hubble_constant_before_the_solve():
    """Its potential is a parameter, and the expansion rate today reads the field."""

    from helpers import QCDM, QCDM_COSMOLOGY, cosmologies_of, qcdm_parameters

    from discoeb.components.gravity import GRHO_PER_OMEGA
    from discoeb.models import LCDM

    flat = QCDM.from_dict(qcdm_parameters())
    lcdm = cosmologies_of(LCDM, QCDM_COSMOLOGY)
    assert flat.V0 == pytest.approx(GRHO_PER_OMEGA * lcdm.omega_de, rel=1e-15)
    assert lcdm.H0 == pytest.approx(QCDM_COSMOLOGY["H0"], rel=1e-12)
    with pytest.raises(TypeError, match="needs a solve"):
        flat.H0
    with pytest.raises(TypeError, match="no dark-energy density"):
        QCDM.from_density_parameters(**QCDM_COSMOLOGY)


def test_a_solve_takes_cosmologies_of_one_model():
    from discoeb.models import LCDM, W0WACDM
    from discoeb.models._model import _batch_of

    planck = LCDM(**BASE_COSMOLOGY)
    batch = _batch_of([planck, planck.replace(omega_c=0.13)])
    assert batch.model == LCDM.spec and batch.model_class is LCDM and len(batch) == 2
    assert batch.cosmologies[1]["omega_c"] == 0.13
    assert len(_batch_of(planck)) == 1
    with pytest.raises(TypeError, match="share a model"):
        _batch_of([planck, W0WACDM(**BASE_COSMOLOGY, w_DE_0=-0.9)])
    with pytest.raises(TypeError, match="non-empty"):
        _batch_of([])
    with pytest.raises(TypeError, match=r"LCDM\(omega_b="):
        _batch_of([BASE_COSMOLOGY])


# --- A type checker reads a model's parameters off its components' declarations.


USER_CODE = '''
from discoeb import LCDM, W0WACDM
from discoeb.components.species import MassiveNeutrinos

good = LCDM(
    omega_b=0.022, omega_c=0.117, omega_de=0.31, A_s=2.1e-9, n_s=0.966,
    Tcmb=2.7255, YHe=0.245, Neff=3.046,
)
dark = W0WACDM(
    omega_b=0.022, omega_c=0.117, omega_de=0.31, A_s=2.1e-9, n_s=0.966,
    Tcmb=2.7255, YHe=0.245, Neff=3.046, w_DE_0=-0.9,
)
moved = good.replace(omega_c=0.13)


class Mine(LCDM, MassiveNeutrinos):
    """A model of the user's: LCDM with massive neutrinos."""


mine = Mine(
    omega_b=0.022, omega_c=0.117, omega_de=0.31, A_s=2.1e-9, n_s=0.966,
    Tcmb=2.7255, YHe=0.245, Neff=3.046, mnu=0.06,
)

LCDM(  # missing
    omega_b=0.022, omega_de=0.31, A_s=2.1e-9, n_s=0.966,
    Tcmb=2.7255, YHe=0.245, Neff=3.046,
)
LCDM(  # foreign
    omega_b=0.022, omega_c=0.117, omega_de=0.31, A_s=2.1e-9, n_s=0.966,
    Tcmb=2.7255, YHe=0.245, Neff=3.046, mnu=0.06,
)
LCDM(  # derived
    omega_b=0.022, omega_c=0.117, omega_de=0.31, A_s=2.1e-9, n_s=0.966,
    Tcmb=2.7255, YHe=0.245, Neff=3.046, grhoc=1.0,
)
LCDM(0.31)  # positional
'''


@pytest.mark.skipif(ty_executable() is None, reason="needs ty (the dev group)")
def test_a_type_checker_sees_the_models_parameters(tmp_path):
    found, output = ty_errors(USER_CODE, tmp_path)
    lines = USER_CODE.splitlines()

    def line_of(marker: str) -> int:
        return next(i for i, text in enumerate(lines, 1) if marker in text)

    assert found >= {
        (line_of("# missing"), "missing-argument"),
        (line_of("# positional"), "too-many-positional-arguments"),
    }, output
    rows = {row for row, _ in found}
    for marker in ("# foreign", "# derived"):
        start = line_of(marker)
        assert any(start <= row <= start + 3 for row in rows), (marker, output)
    # and nothing else: the valid calls and the user's model class pass
    assert all(row >= line_of("# missing") for row in rows), output


# --- The Jacobian sparsity pattern, and what modax's analysis makes of it.


ORDER_ONE = dict(a=0.5, opacity=3.0, cs2=0.3)
"""Thermal channels of order one: the finite differences below must resolve
every coupling, and at a realistic ``cs2 ~ 1e-8`` or ``a ~ 1e-3`` the baryon
pressure and the dark-energy density fall below the rounding of the rest of
the row."""


def finite_difference_jacobian_pattern(
    model, seed=0, tau=100.0
) -> set[tuple[int, int]]:
    """The nonzeros of the host-built right-hand side's Jacobian at a random state and ``tau``."""
    built = model.build(ConstantThermalHistory(**ORDER_ONE))
    rhs = built.build(PARAM_COLUMNS, **HOST)
    rng = np.random.default_rng(seed)
    p = rng.uniform(
        0.5, 1.5, N_PARAM
    )  # order one, curvature and sensitivity direction included
    p[PARAM_COLUMNS["w_DE_0_nominal"]] = -0.9
    y = rng.standard_normal(nvar(model))

    def f(state):
        return np.asarray(rhs(tuple(state), tau, tuple(p)))

    jacobian = np.zeros((nvar(model), nvar(model)))
    for j in range(nvar(model)):
        e = np.zeros(nvar(model))
        e[j] = 1e-6
        jacobian[:, j] = (f(y + e) - f(y - e)) / 2e-6
    # A structural zero is exactly zero: the variable never enters the expression.
    return entries(zip(*np.nonzero(np.abs(jacobian) > 1e-9 * np.abs(jacobian).max())))


@pytest.mark.parametrize("name", list(MODELS))
def test_the_pattern_is_the_jacobians(name):
    """Traced through the providers, the pattern is exactly the nonzeros of the Jacobian.

    Of both regimes of the radiation streaming approximation: at ``tau = 1``
    every mode is outside ``k tau = 45`` and the hierarchies stream, at
    ``tau = 100`` (``a = 0.5``, past the latch) they are replaced and frozen.
    The pattern is their union, which a colouring needs, and a few entries
    more: the baryon velocity and the photon dipole read ``z`` through the
    slip's ``rsa * (rsa_q - qg)`` while the slip itself is scaled by ``1 -
    rsa``, a coefficient zero in both regimes that a tracer cannot see
    through, so those two rows carry the extra, safe superset.
    """
    model = MODELS[name]
    pattern = model.sparsity()

    assert pattern.ndim == 2 and pattern.shape[1] == 2 and pattern.dtype == np.int64
    assert pattern.min() >= 0 and pattern.max() < nvar(model)
    streaming = finite_difference_jacobian_pattern(model, tau=1.0)
    approximated = finite_difference_jacobian_pattern(model, tau=100.0)
    assert streaming | approximated <= entries(pattern)
    extra = entries(pattern) - (streaming | approximated)
    assert {r for r, _ in extra} <= {state_index(model)[n] for n in ("vb", "theta_1")}
    assert approximated < entries(pattern), (
        "the approximation freezes the hierarchies' rows"
    )
    assert not diagonal(model) <= entries(pattern), (
        "an undamped multipole has no diagonal"
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_the_pattern_has_the_physics_structure(name):
    """A core of metric, fluids and the lowest multipoles, bordered by tridiagonal free-streaming tails.

    This is the structure the states used to declare and the sparse solver
    exploits; the trace finds it in the equations, and finds the core far from
    dense: a free-streaming multipole reads its two neighbours (and itself,
    when Thomson scattering damps it), and the metric reads the densities and
    momenta but not the shears.
    """
    model = MODELS[name]
    found = entries(model.sparsity())
    core, tails = structure(model)

    assert found <= dense_core_pattern(model)
    for tail in tails:
        for i, idx in enumerate(tail):
            neighbours = {tail[i - 1] if i else tail[0] - 1, idx}
            if i + 1 < len(tail):
                neighbours.add(tail[i + 1])
            assert {c for r, c in found if r == idx} <= neighbours, f"{name}: row {idx}"
            assert (idx, tail[i - 1] if i else tail[0] - 1) in found, (
                f"{name}: row {idx}"
            )
    # Every variable couples to something and something couples to it: a
    # state index the pattern never mentions is one the sparse solver would
    # silently drop.
    assert {r for r, _ in found} == {c for _, c in found} == set(range(nvar(model)))
    # Nothing in the core is dense: the sums reach the metric, not every row.
    assert len({(r, c) for r, c in found if r in core and c in core}) < len(core) ** 2


def test_the_flat_lcdm_pattern_is_the_expected_size():
    """50 variables, 173 entries.

    The metric reads ``z`` (``etak`` and the four densities) and ``dgq`` (the
    three momenta): 8. CDM reads ``z``: 5; the baryon density ``z`` and
    ``vb``: 6; ``vb`` itself, ``clxb`` and the dipole through the slip: 3. The
    photon monopole ``z`` and the dipole: 6; the dipole its neighbours and the
    slip: 4; the quadrupole its neighbours, ``sigma`` (``z`` and ``dgq``),
    itself and ``E_2``: 11; the 12 damped free-streaming multipoles 3 each and
    the last 2. ``E_2`` reads ``E_3``, itself and the quadrupole: 3; the E-mode
    tail 3 each and 2 at the end. The neutrino monopole 6, the dipole 2, the
    quadrupole its neighbours and ``sigma``: 9; the undamped tail 2 each.
    The radiation streaming approximation adds 8: under it the photon
    momentum in the slip is ``-4 z / 3``, so the baryon velocity and the
    photon dipole also read ``z``'s ``etak``, ``clxc`` and the monopole or
    density they did not already: 4 each.
    """
    model = MODELS["flat-lcdm"]
    assert nvar(model) == 50
    assert (
        len(model.sparsity())
        == (
            8
            + 5
            + 6
            + 3
            + (6 + 4 + 11 + 12 * 3 + 2)
            + (3 + 12 * 3 + 2)
            + (6 + 2 + 9 + 12 * 2 + 2)
            + 8
        )
        == 173
    )


# --- A species with a parameter of its own, added without touching the package.


def test_the_components_check_the_parameter():
    cosmologies_of(LCDM_X, _split(COSMOLOGY))
    cosmologies_of(LCDM, COSMOLOGY)
    cosmologies_of(LCDM, _split(COSMOLOGY, 0.0))  # at its default, the key is harmless
    with pytest.raises(ValueError, match="must not be negative"):
        cosmologies_of(LCDM_X, _split(COSMOLOGY, -0.01))
    with pytest.raises(ValueError, match="use a model with SecondDarkMatter"):
        cosmologies_of(LCDM, _split(COSMOLOGY))
