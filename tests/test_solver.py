"""Tests of ``discoeb.solver``, one section per module."""

import numpy as np
import pytest
from helpers import (
    COSMOLOGY,
    DENSITY_PARAMETERS,
    DIAG_A,
    DIAG_KEEP,
    KERNELS_BUILT,
    LAM,
    LCDM_OUTSIDE,
    LCDM_T,
    LCDM_X,
    LCDM_XY,
    MODELS,
    OMEGA_X,
    OUTSIDE_KEEP,
    PK_KEEP,
    QCDM,
    QCDM_COSMOLOGY,
    QCDM_KEEP,
    SOLVE_EB_K,
    _matter_power,
    _more_parameter_case,
    _perturbed_cosmologies,
    _split,
    by_name,
    cosmologies_of,
    count_kernels,
    dense_core_pattern,
    diagonal,
    entries,
    model_parameters,
    nvar,
    on_the_host,
    packed,
    physical,
    qcdm_parameters,
    structure,
)

from discoeb.components._component import (
    Interaction,
    derivative,
)
from discoeb.components.recombination import ReionizationDepth
from discoeb.components.species import (
    Fluid,
)
from discoeb.models import LCDM
from discoeb.solver._kernel import (
    A_START,
    joint_kernel,
    joint_model,
    thermal_indices,
)

# ============================================================================
# kernel: The joint kernel (``_kernel``): the two blocks, the parameter row, the initial state and the linear solver's layout
# ============================================================================

# --- One component listed several times, each copy under a name of its own.


def test_two_copies_share_a_model():
    from discoeb.solver._kernel import parameter_layout

    keys = LCDM_XY.parameter_keys()
    assert {"x_omega", "x_w", "x_cs2", "y_omega", "y_w", "y_cs2"} <= set(keys)
    assert "omega" not in keys and "x_omega" not in LCDM.parameter_keys()
    names = LCDM_XY.state_names()
    assert names[-4:] == ("x_delta", "x_theta", "y_delta", "y_theta")
    layout = parameter_layout(LCDM_XY)
    # the parameters first, the recombination block's last of them, then
    # what the host computes from parameters
    names = list(layout.columns)
    assert names[names.index("z_re") + 1 :] == [
        "K",
        "omega_g",
        "omega_ur",
        "fHe",
        "akthom",
        "Nnow",
    ]
    assert {"x_w", "y_cs2", "x_omega", "y_omega"} <= set(layout.columns)
    # unnamed, the fluid's cs2 would be the baryons'
    with pytest.raises(ValueError, match="needs a name"):
        LCDM.add(Fluid, name="clash").build()


# --- The Einstein-Boltzmann component model, on the host.


def test_the_initial_state_is_the_models():
    _, y0 = packed(LCDM, cosmologies_of(LCDM, COSMOLOGY), [1e-2, 1e-1])
    assert y0.shape == (2, 55) and np.isfinite(y0).all()
    np.testing.assert_allclose(y0[:, 0], -1.0, rtol=1e-3)  # eta ~ -1
    # the thermal state: the conformal time of the radiation era, then RECFAST's
    assert np.all((0.05 < y0[:, 50]) & (y0[:, 50] < 0.2))
    np.testing.assert_allclose(
        y0[:, 51:], [[1.0, 2.0, COSMOLOGY["T_cmb"] / A_START, 0.0]] * 2
    )


# --- The Jacobian sparsity pattern, and what modax's analysis makes of it.


sparse_direct = pytest.importorskip("modax._sparse_direct")


colour_sparsity = pytest.importorskip("modax._sparsity").colour_sparsity


normalize_sparsity = pytest.importorskip("modax._sparsity").normalize_sparsity


@pytest.mark.parametrize("name", list(MODELS))
def test_amd_fills_in_only_the_core(name):
    """The exact pattern factorises in fewer slots than the dense core ever took.

    Minimum degree still peels each tail from its truncated end, where every
    variable has degree two, so the tails fill in not at all; the core is no
    longer a clique, so eliminating it creates some fill, but never beyond
    the dense core the pattern used to declare -- which factorised with zero
    fill, so its size is exactly what the solver used to work on.
    """
    model = MODELS[name]
    pattern = normalize_sparsity(model.sparsity(), nvar(model))
    layout_lu = sparse_direct.analyse(pattern, "amd")
    declared = dense_core_pattern(model)
    core, _ = structure(model)

    nnz = sum(len(cols) for cols in pattern)
    assert nnz < layout_lu.nnz < len(declared)
    table = layout_lu.slot_table()
    factorised = entries(zip(*np.nonzero(table >= 0)))
    # The factorisation holds every diagonal, for the ``1/(h gamma)``.
    fill = factorised - entries(model.sparsity()) - diagonal(model)
    assert fill and fill <= {(r, c) for r in core for c in core}
    assert factorised <= declared


def test_the_solver_and_the_kernel_share_one_layout():
    """The slots the sweeps write are the slots the factorisation reads; the fill has its own."""
    model = MODELS["flat-lcdm"]
    solver = sparse_direct.sparse_direct_solver(model.sparsity(), nvar(model))
    pattern = normalize_sparsity(model.sparsity(), nvar(model))

    assert (
        solver.compressed.n_colours == 11
    )  # 12 Enzyme sweeps, not 51; the dense core needed 13
    assert (
        solver.compressed.size == solver.nnz == 203
    )  # 173 entries, 14 undeclared diagonals, 16 of fill
    slots = [
        solver.compressed.slot(r, c) for r, cols in enumerate(pattern) for c in cols
    ]
    assert len(set(slots)) == len(slots) == 173, "every entry has a slot of its own"
    assert set(slots) <= set(range(203))


@pytest.mark.parametrize("name", list(MODELS))
def test_undeclared_diagonals_get_the_factorisations_slots(name):
    """Where ``J`` has no diagonal, the ``1/(h gamma)`` must still reach the factorisation's.

    An undamped multipole's column shares a colour with a neighbour that has
    an entry in its row, so the ``(row, colour)`` grid cell of its diagonal
    belongs to that neighbour. modax used to resolve ``slot(i, i)`` through
    the grid and put the ``1/(h gamma)`` there, for twelve rows of the
    default model, and the solve returned the initial state; it now goes
    through the factorisation's own diagonal table (modax 9f54d12).
    """
    model = MODELS[name]
    pattern = normalize_sparsity(model.sparsity(), nvar(model))
    table = sparse_direct.analyse(pattern, "amd").slot_table()
    compressed = sparse_direct.sparse_direct_solver(
        model.sparsity(), nvar(model)
    ).compressed

    undeclared = [i for i in range(nvar(model)) if i not in pattern[i]]
    taken = [
        i
        for i in undeclared
        if compressed.packed[i * compressed.n_colours + compressed.colour[i]] >= 0
    ]
    assert taken, "the case the grid lookup got wrong"
    assert all(compressed.slot(i, i) == table[i, i] for i in range(nvar(model)))
    assert compressed.diagonal == tuple(table[i, i] for i in range(nvar(model)))


@pytest.mark.parametrize("name", list(MODELS))
def test_the_exact_pattern_needs_no_more_colours_than_the_dense_core(name):
    model = MODELS[name]
    exact = colour_sparsity(normalize_sparsity(model.sparsity(), nvar(model)))
    declared = colour_sparsity(
        normalize_sparsity(np.array(sorted(dense_core_pattern(model))), nvar(model))
    )
    assert exact.n_colours < declared.n_colours


def test_the_colouring_keeps_every_entry_apart():
    """Two columns of one colour sharing a row would share a slot.

    modax checks this when it colours; asserting it here is what says the
    *pattern* is the reason it holds, rather than luck in the colouring.
    """
    for name, model in MODELS.items():
        pattern = normalize_sparsity(model.sparsity(), nvar(model))
        compressed = colour_sparsity(pattern)
        for row, cols in enumerate(pattern):
            colours = {compressed.colour[c] for c in cols}
            assert len(colours) == len(cols), f"{name}: row {row} collides"


# --- A species with a parameter of its own, added without touching the package.


def test_the_parameter_reaches_the_dictionary_the_row_and_the_inputs():
    from discoeb.solver._kernel import parameter_layout

    assert LCDM_X.parameter_keys()["omega_x"] == 0.0
    assert "omega_x" not in LCDM.parameter_keys()
    layout = parameter_layout(LCDM_X)
    own = parameter_layout(LCDM).columns
    assert sorted(layout.columns) == sorted((*own, "omega_x"))
    # the wave number leads the row, then the parameters in component order
    assert list(layout.columns)[0] == "k"
    assert sorted(layout.columns.values()) == list(range(layout.n))
    assert dict(layout.values) == {n: c for n, c in layout.columns.items() if n != "k"}
    # The built-in models' rows have no trace of it.
    assert "omega_x" not in parameter_layout(LCDM).columns


# --- The joint system as a modax hierarchy: the recombination block publishing what the hierarchy reads.


def test_the_recombination_block_publishes_the_thermal_quantities():
    colour_sparsity = pytest.importorskip("modax._sparsity").colour_sparsity

    nvar = len(LCDM.state_names())
    model, hierarchy = joint_kernel(LCDM)
    assert model.nvar == nvar + 5
    perturbations, recombination = hierarchy.blocks[-1], hierarchy.blocks[0]
    assert hierarchy.layout == (
        perturbations,
        recombination,
    )  # the thermal states stay last
    # nothing declares the split: the trace finds the states that read no
    # wave number and the fewest quantities the hierarchy needs of them
    background, others = model.stages()
    assert background == ("tau", "thermo", "kappa")
    # the scale factor is the time itself, and the conformal time a state
    assert model.published() == ("tau", "opacity", "cs2")
    assert recombination.n == 5 and recombination.n_outputs == 3
    # the reionization depth is a background state a model adds
    with_depth = joint_model(LCDM.add(ReionizationDepth, name="LCDM+reio"))
    assert with_depth.stages()[0] == ("kappa_reio", "tau", "thermo", "kappa")
    assert with_depth.published() == model.published()
    assert perturbations.n == nvar and perturbations.reads == (recombination,)
    assert model.state_indices(background) == list(thermal_indices(model))
    # the hierarchy alone, as the model's sparsity traces it: 173 entries
    # (165 and the slip's reading of h' under the radiation streaming approximation), 11 colours
    main = (
        perturbations.sparsity
    )  # normalised by the block: the sorted columns of each row
    assert sum(len(cols) for cols in main) == 173
    assert colour_sparsity(main).n_colours == 11


@pytest.mark.gpu
def test_the_recombination_block_steps_on_its_own():
    """The recombination block takes the same, fewer steps for every wave mode of a cosmology.

    On the history kernel ``test_solve_eb_keeps_what_the_history_holds`` builds.
    """
    import numpy as np
    from helpers import _perturbed_cosmologies

    from discoeb.solver._kernel import _solve_batch

    k_values = np.array([1.0e-3, 1.0e-2, 1.0e-1, 1.0])
    (_, stats), _, _ = _solve_batch(
        LCDM,
        tuple(cosmologies_of(LCDM, _perturbed_cosmologies(1))),
        k_values,
        return_stats=True,
    )
    accepted, rejected, loop = (
        np.asarray(stats[k]) for k in ("accepted_steps", "rejected_steps", "loop_steps")
    )
    driver = loop - accepted - rejected
    # the thermal history does not know the wave number, but the error the
    # hierarchy inherits from it does, so the block's steps vary with k
    assert 50 < driver.min() and driver.max() < 500, driver
    # ... while the hierarchy's own steps, with the radiation streaming
    # approximation cutting its late oscillations, range far more widely: the
    # block is not stepping with it
    assert np.ptp(driver) * 4 < np.ptp(accepted), (driver, accepted)
    assert accepted[-1] > accepted[0], (
        accepted
    )  # and the hierarchy steps more at high k


# --- A background state added from outside: the equations split the blocks themselves.


class Ageing(Interaction):
    """A perturbation equation that reads the background state itself."""

    @derivative("delta_c")
    def d_delta_c(self, t):
        return 0.0 * t


def test_a_state_that_reads_no_wave_number_joins_the_background():
    model = joint_model(LCDM_T)
    background, perturbations = model.stages()
    assert background == ("t", "tau", "thermo", "kappa")
    assert "t" not in perturbations and "delta_c" in perturbations
    # nobody reads it, so the perturbations are handed what LCDM's are
    assert model.published() == ("tau", "opacity", "cs2")
    # without the recombination block the scale factor is an input
    assert LCDM_T.build().stages()[0] == ("t",)
    assert LCDM.build().stages()[0] == ()


def test_a_background_state_the_perturbations_read_is_published():
    model = joint_model(LCDM_T.add(Ageing, name="LCDM+t+ageing"))
    assert model.published() == ("t", "tau", "opacity", "cs2")


# --- A scalar field from outside the package: background states, and a budget closed by shooting.


def test_the_field_is_background_and_its_perturbation_is_not():
    model = joint_model(QCDM)
    background, perturbations = model.stages()
    assert background == ("phi", "phidot", "tau", "thermo", "kappa")
    assert {"dphi", "ddphi"} <= set(perturbations)
    # the perturbations read the field's velocity itself, and the potential
    # stands for the field: its slope and curvature follow from it
    assert model.published() == ("phidot", "tau", "V", "opacity", "cs2")


# ============================================================================
# sensitivities: modax's forward sensitivities: derivatives of ``solve_eb`` values against central differences
# ============================================================================

# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


@pytest.mark.gpu
def test_solve_eb_derivatives(two_columns):
    """The kept quantities are differentiable in the cosmological parameters.

    The derivative chains the densities' dependence on ``Omega_m``, ``Omega_b``,
    ``H_0``, ``T_cmb`` and ``N_eff`` into the parameter row, along whose
    tangent modax integrates sensitivity columns and stores each quantity's
    derivative: ``jax.jvp`` and ``jax.jacfwd`` of the kept values come from
    the joint launch, and reverse mode raises. The derivative in each
    column of the row (``inputs``) is checked against central differences
    of its own primal, at fixed initial state;
    the differences are the noisy side: the solve runs at rtol = atol = 1e-4,
    a step-acceptance flip between the two shifted solves moves a value by
    ~1e-6 of its scale, so the Jacobian is checked as the change it predicts
    over a 0.1 percent step against the measured change, in units of each
    quantity's scale. The parameter-level derivative moves the row and the
    initial state, and is checked the same way.

    The two contrasts ``P(k)`` reads are kept today, in launches of two
    columns: the kernels of the ``P(k)`` derivative tests, which are
    compiled per kept name and save time. The hook's tangent is the same
    code for any quantity and time. (Until 2026-11-17 this kept ``R`` and
    the photon quadrupole too, at twelve scale factors, on three kernels of
    its own.)
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM
    from discoeb.solver._solver import solve_eb, solve_inputs

    DIFFERENTIABLE_PARAMETERS = DENSITY_PARAMETERS
    cosmology = _perturbed_cosmologies(1)[0]
    # the inputs exactly as solve_eb forms them: a last-bit difference moves a
    # step-acceptance and with it a value by ~1e-6
    theta = jnp.asarray([cosmology[key] for key in DIFFERENTIABLE_PARAMETERS])
    fixed = {k: v for k, v in cosmology.items() if k not in DIFFERENTIABLE_PARAMETERS}

    def inputs_of(theta):
        param = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta))}
        return solve_inputs(LCDM.from_dict(physical(LCDM, param)))

    densities = inputs_of(theta)
    a = np.array([1.0])
    names = PK_KEEP
    noise = {"delta_c": 5.0e-6, "delta_b": 5.0e-6}

    # converted from Omega_m and H_0 as JAX values, as inputs_of converts them
    traced = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta))}
    plain = solve_eb(cosmologies_of(LCDM, traced), SOLVE_EB_K, a, keep=names)
    floats = cosmologies_of(LCDM, cosmology)

    def kept(densities, **settings):
        return solve_eb(floats, SOLVE_EB_K, a, keep=names, inputs=densities, **settings)

    base = np.asarray(kept(densities))
    # The same launch, from densities formed in JAX rather than as floats.
    np.testing.assert_allclose(np.asarray(plain), base, rtol=1e-12)
    scale = np.abs(base).max(axis=(2, 3), keepdims=True)

    # The joint launch's primal: the same trajectories to within the
    # solver's tolerance.
    still = {name: jnp.zeros_like(value) for name, value in densities.items()}
    primal, _ = jax.jvp(kept, (densities,), (still,))
    assert np.abs(np.asarray(primal) - base).max() < 1.0e-4 * scale.max()

    # The central differences are converged (rtol = atol = 1e-6): at the default
    # tolerance the solve's own error, different on the two sides of a step,
    # is the larger part of a difference.
    converged = dict(rtol=1.0e-6, atol=1.0e-6, max_steps=200_000)

    def check(jac, finite, step, what):
        err = np.abs(jac - finite) * step / scale
        for q, name in enumerate(names):
            worst = err[0, q].max()
            assert worst < noise[name], (
                f"d{name}/d{what}: {worst:.3g} of the scale over a 1e-3 step"
            )

    # One direction per column of the row, in launches of two.
    jac = jax.jacfwd(kept)(densities)
    assert set(jac) == set(densities)
    for dn, column in jac.items():
        column = np.asarray(column)[..., 0]
        assert column.shape == base.shape and np.isfinite(column).all()
        if densities[dn][0] == 0.0:  # omega_k and z_re of a flat LCDM
            continue
        step = 1.0e-3 * float(densities[dn][0])
        finite = (
            np.asarray(kept({**densities, dn: densities[dn] + step}, **converged))
            - np.asarray(kept({**densities, dn: densities[dn] - step}, **converged))
        ) / (2.0 * step)
        check(column, finite, step, dn)

    # Through the cosmology: one column per parameter, along the tangent of
    # the row and of the initial state.
    def of_parameters(theta, **settings):
        param = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta))}
        return solve_eb(
            cosmologies_of(LCDM, param), SOLVE_EB_K, a, keep=names, **settings
        )

    jac_theta = np.asarray(jax.jacfwd(of_parameters)(theta))
    assert np.all(np.abs(jac_theta).max(axis=(0, 1, 2, 3)) > 0.0)
    for p, key in enumerate(DIFFERENTIABLE_PARAMETERS):
        step = 1.0e-3 * float(theta[p])
        finite = (
            np.asarray(of_parameters(theta.at[p].add(step), **converged))
            - np.asarray(of_parameters(theta.at[p].add(-step), **converged))
        ) / (2.0 * step)
        check(jac_theta[..., p], finite, step, key)

    # Forward mode only: modax integrates along the tangents it is given.
    with pytest.raises(NotImplementedError, match="forward mode"):
        jax.grad(lambda t: jnp.sum(of_parameters(t)))(theta)

    # Every key is differentiable, Y_He too (one direction: the kernel above).
    def of_helium(Y_He):
        batch = cosmologies_of(LCDM, {**cosmology, "Y_He": Y_He})
        return solve_eb(batch, SOLVE_EB_K, a, keep=names)

    _, along = jax.jvp(
        of_helium, (jnp.asarray(cosmology["Y_He"]),), (jnp.asarray(1.0),)
    )
    assert (
        np.isfinite(np.asarray(along)).all() and np.abs(np.asarray(along)).max() > 0.0
    )


REIO_NAMES = ("R", "delta_b", "kappa", "kappa_reio")


@pytest.mark.slow
@pytest.mark.gpu
def test_reionization_derivative():
    """``solve_eb``'s ``jax.jvp`` in ``z_re`` against central differences.

    The differences come from converged solves (rtol = atol = 1e-6), the
    sensitivity from one at the default tolerance, as in
    :func:`test_solve_eb_derivatives`. ``kappa``, the optical depth, and
    ``kappa_reio``, which is ``tau_reio`` today, carry the reionization
    redshift directly. One of the kept quantities' derivatives in a
    parameter beyond the densities': until 2026-11-17 ``w_DE_0``,
    ``w_DE_a``, ``Omega_k`` and ``m_nu`` had this test too, on two kernels
    a model, and they keep ``test_more_parameters_matter_power``.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.solver._solver import solve_eb

    name = "z_re"
    model, cosmology, step, keep = _more_parameter_case(name)
    a = DIAG_A
    kept = [keep.index(quantity) for quantity in REIO_NAMES]

    def values(p, **options):
        batch = cosmologies_of(model, {**cosmology, name: p})
        return solve_eb(batch, SOLVE_EB_K, a, keep=keep, **options)[:, kept]

    p0 = jnp.asarray(float(cosmology[name]))
    _, tangent = jax.jvp(values, (p0,), (jnp.ones_like(p0),))
    tangent = np.asarray(tangent)
    converged = dict(rtol=1.0e-6, atol=1.0e-6, max_steps=200_000)
    finite = (
        np.asarray(values(p0 + step, **converged))
        - np.asarray(values(p0 - step, **converged))
    ) / (2.0 * step)
    assert np.isfinite(tangent).all()
    value = np.abs(np.asarray(values(p0)))
    # While the step is under way a difference over a unit of z_re, itself
    # twice the step's width, is not the derivative (15 percent off for
    # kappa_reio), so the depth is compared once every step is over, z < 3.
    settled = DIAG_A > 0.25
    for q, quantity in enumerate(REIO_NAMES):
        at = settled if quantity == "kappa_reio" else slice(None)
        scale = np.abs(value[0, q][:, at]).max()
        err = np.abs(tangent[0, q][:, at] - finite[0, q][:, at]).max()
        signal = np.abs(finite[0, q][:, at]).max()
        print(
            f"d{quantity}/d{name}: {err / signal:.3g} of the largest difference, "
            f"which moves the quantity by {signal * step / scale:.3g} of its scale"
        )
        if signal * step < 1.0e-4 * scale:
            # A change the solve's noise swamps: the sensitivity must predict
            # one as small.
            assert err * step < 1.0e-5 * scale, f"d{quantity}/d{name}"
            continue
        assert err < 1.0e-3 * signal, (
            f"d{quantity}/d{name}: {err / signal:.3g} of its largest central difference"
        )


# ============================================================================
# solver: ``solve_eb`` (``_solver``): kept quantities, and solves of models with components added in the tests
# ============================================================================

# --- One component listed several times, each copy under a name of its own.


OMEGA_FLUID = 0.04
"""The share of ``Omega_m`` the named fluid holds, beside ``OMEGA_X`` in the second dark matter."""


def test_a_dictionary_names_no_model():
    """No model stands in for a bare dictionary: the caller names one."""

    from discoeb import solve_inputs
    from discoeb.solver._kernel import parameter_layout

    with pytest.raises(TypeError, match="LCDM"):
        solve_inputs(COSMOLOGY)
    inputs = solve_inputs(cosmologies_of(LCDM, COSMOLOGY))
    assert ("k", *inputs) == tuple(parameter_layout(LCDM).columns)
    assert all(value.shape == (1,) for value in inputs.values())


SOLVE_K = np.geomspace(1.0e-3, 0.3, 8)


SOLVE_A = DIAG_A


@pytest.mark.slow
@pytest.mark.gpu
def test_components_from_outside_leave_the_solve_unchanged():
    """A third of the CDM in a species of the tests' own and a third in a named fluid evolve as the CDM did.

    One model holds every kind of component a user adds
    (:data:`helpers.LCDM_OUTSIDE`): ``SecondDarkMatter``, with a column
    parameter of its own, a named ``Fluid`` and the background state
    ``CosmicTime``. Both fluids' contrasts equal the CDM's in the same solve,
    the contrast, the curvature perturbation and the conformal time agree
    with the plain LCDM solve to the solver's tolerance, the cosmic time ends
    at the age of the universe, and the input ``omega_x`` moves
    every kept quantity as ``omega_c`` does. (Three tests and seven kernels
    until 2026-11-17, one model each.)
    """

    import jax
    import jax.numpy as jnp

    from discoeb import solve_eb, solve_inputs

    lcdm = cosmologies_of(LCDM, COSMOLOGY)
    plain = by_name(solve_eb(lcdm, SOLVE_K, SOLVE_A, keep=DIAG_KEEP), DIAG_KEEP)
    batch = cosmologies_of(
        LCDM_OUTSIDE,
        {
            **_split(COSMOLOGY),
            "Omega_m": COSMOLOGY["Omega_m"] - OMEGA_X - OMEGA_FLUID,
            "y_Omega": OMEGA_FLUID,
        },
    )
    keep = OUTSIDE_KEEP
    split = by_name(solve_eb(batch, SOLVE_K, SOLVE_A, keep=keep), keep)

    # The same expansion: the conformal time, a state, comes out the same.
    # From a = 1e-6 on: the closed form it starts from counts the matter of
    # ``rho_matter``, to which a ``Fluid`` does not contribute, so it starts
    # 2e-6 Mpc apart, 2e-5 of itself at ``A_START``.
    late = SOLVE_A >= 1.0e-6
    np.testing.assert_allclose(
        split["tau"][..., late], plain["tau"][..., late], rtol=1e-5
    )
    # The same equations and initial state; the fp32 LU tells the columns apart.
    np.testing.assert_allclose(split["delta_x"], split["delta_c"], rtol=1e-5)
    np.testing.assert_allclose(split["y_delta"], split["delta_c"], rtol=1e-5)
    # The initial states agree (the growing mode's om sums rho_matter, which
    # both fluids contribute to), so what is left is the solver's: the
    # smooth quantities to 1e-3, the oscillating photon quadrupole, stepped
    # differently by a larger system, to a few times that.
    for name, gate in (("delta_c", 1e-3), ("R", 1e-3), ("F_2", 5e-3)):
        ours, theirs = split[name], plain[name]
        assert np.abs(ours - theirs).max() < gate * np.abs(theirs).max(), name

    # t = int a dtau = int a / aH dln a, by the trapezoid rule on the host
    grid = np.geomspace(1.0e-9, 1.0, 20000)
    rate = np.asarray(on_the_host(LCDM, COSMOLOGY, "aH", a=grid))
    age = np.trapezoid(grid / rate, np.log(grid))
    conformal_age = np.trapezoid(1.0 / rate, np.log(grid))
    assert 4000.0 < age < 4500.0  # 13.8 Gyr in Mpc
    # the background block carries its states to today for every wave mode
    np.testing.assert_allclose(split["t"][0, :, -1], age, rtol=1e-4)
    np.testing.assert_allclose(split["tau"][0, :, -1], conformal_age, rtol=1e-4)

    x = solve_inputs(batch)

    def along(name):
        # one direction, so the model's one-column kernel
        def kept(value):
            return solve_eb(batch, SOLVE_K, SOLVE_A, keep=keep, inputs={name: value})

        return np.asarray(jax.jvp(kept, (x[name],), (jnp.ones_like(x[name]),))[1])

    along_c, along_x = along("omega_c"), along("omega_x")
    assert np.isfinite(along_c).all() and np.isfinite(along_x).all()
    scale = np.abs(along_c).max(axis=(-2, -1), keepdims=True)
    moved = scale[0, :, 0, 0] > 0.0
    assert moved[[keep.index(name) for name in ("delta_c", "R", "F_2", "tau")]].all()
    worst = (np.abs(along_x - along_c) / np.where(scale > 0.0, scale, 1.0)).max()
    assert worst < 1e-3, worst


# --- A scalar field from outside the package: background states, and no budget to close.


FIELD_KEEP = {LCDM: DIAG_KEEP, QCDM: QCDM_KEEP}
"""The names each model's solves keep in the scalar-field test: one kernel a model."""


def _power(model, cosmology, k):
    return np.asarray(
        _matter_power(
            k, cosmologies_of(model, cosmology), a=DIAG_A, keep=FIELD_KEEP[model]
        )
    )[0]


GROWTH_A = DIAG_A[DIAG_A >= 1.0e-5]
"""The part of the shared save grid the growth factor is integrated over."""


def _growth(model, cosmology):
    """``(growth factor, H_0)`` in the model's background, whose expansion rate the solve itself gives.

    The matter growth factor today, ``delta'' + aH delta' = 4 pi G a^2 rho_m
    delta``, by Heun steps in ``ln a`` from ``a = 1e-5`` with ``aH`` kept
    on that grid. Only the ratio between two models that agree early on is
    meaningful. ``H_0`` in km/s/Mpc is the rate today.
    """

    from discoeb import solve_eb
    from discoeb.components.gravity import H100, H100_SQ

    cosmology = cosmologies_of(model, cosmology)
    a = GROWTH_A
    keep = FIELD_KEEP[model]
    rate = np.asarray(solve_eb(cosmology, [1.0e-2], DIAG_A, keep=keep))
    rate = rate[0, keep.index("aH"), 0, DIAG_A >= 1.0e-5]
    d = model_parameters(model, cosmology)
    source = 1.5 * H100_SQ * float(d["omega_c"] + d["omega_b"]) / a
    lna = np.log(a)
    delta, slope = 1.0, 0.0
    for i in range(len(a) - 1):
        h = lna[i + 1] - lna[i]
        f0 = (slope / rate[i], (-rate[i] * slope + source[i] * delta) / rate[i])
        d1, s1 = delta + h * f0[0], slope + h * f0[1]
        f1 = (
            s1 / rate[i + 1],
            (-rate[i + 1] * s1 + source[i + 1] * d1) / rate[i + 1],
        )
        delta += 0.5 * h * (f0[0] + f1[0])
        slope += 0.5 * h * (f0[1] + f1[1])
    return delta, float(rate[-1]) * 100.0 / H100


@pytest.mark.slow
@pytest.mark.gpu
def test_the_solve_matches_lcdm_and_the_growth_in_the_fields_background():
    """A flat potential gives LCDM, and a rolling field changes ``P(k)`` as its expansion history says.

    The field's potential is a parameter and no budget closes, so its Hubble
    constant comes out of the solve: at LCDM's height a rolling field has
    lost density by today and expands more slowly. Inside the horizon a
    field with unit sound speed does not cluster, so the power relative to
    LCDM is the squared ratio of the growth factors in the two backgrounds,
    integrated here on the host from the expansion rate the solve gives.
    """

    k = np.geomspace(1.0e-3, 0.3, 8)
    lcdm = _power(LCDM, QCDM_COSMOLOGY, k)
    flat = _power(QCDM, qcdm_parameters(), k)
    np.testing.assert_allclose(flat, lcdm, rtol=2e-3)

    rolling = qcdm_parameters(lam=LAM)
    field = _power(QCDM, rolling, k)
    growth_lcdm, h0_lcdm = _growth(LCDM, QCDM_COSMOLOGY)
    growth_field, h0_field = _growth(QCDM, rolling)
    assert h0_lcdm == pytest.approx(QCDM_COSMOLOGY["H_0"], rel=1e-4)
    assert 0.9 * h0_lcdm < h0_field < h0_lcdm
    # the field is background: what the hook reads of it does not know the
    # wave number (it did until 2026-11-17, when the joint state was not laid
    # out as the solver's two blocks are and the hook read the perturbation)
    from discoeb import solve_eb

    field_states = [QCDM_KEEP.index(name) for name in ("phi", "phidot")]
    kept = np.asarray(
        solve_eb(cosmologies_of(QCDM, rolling), k, DIAG_A, keep=QCDM_KEEP)
    )[0, field_states, :, -1]
    np.testing.assert_allclose(kept, kept[:, :1] * np.ones(len(k)), rtol=1e-6)
    assert np.all(kept > 0.0)
    ratio = field / lcdm
    growth = (growth_field / growth_lcdm) ** 2
    assert abs(growth - 1.0) > 3e-3  # the field's history is not LCDM's
    np.testing.assert_allclose(ratio[k > 0.01], growth, rtol=1e-3)
    # near the horizon the field's own perturbation takes part
    assert abs(ratio[0] - ratio[-1]) > 3e-3


# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


SOLVE_EB_A = DIAG_A[1:]
"""After the start, which ``solve_eb`` then adds and drops: the shared grid's 2000 saves."""


@pytest.mark.gpu
def test_solve_eb_keeps_what_the_history_holds():
    """The hooked launch records exactly what the full history evaluates to.

    State elements are copied from the same dense-output states the history
    launch saves; the derived quantities (the comoving curvature perturbation,
    the shear, the expansion rate) are the components' equations compiled into
    the hook, checked here against the same equations evaluated on the host
    over the saved history. The last scale factor is today, so the CDM and
    baryon contrasts there give the batch's ``P(k)``.
    """

    from discoeb.components.recombination import THERMO_ERROR_WEIGHT
    from discoeb.models import LCDM
    from discoeb.solver._kernel import A_START, _solve_batch, parameter_layout
    from discoeb.solver._solver import solve_eb

    cosmologies = _perturbed_cosmologies(2)
    names = DIAG_KEEP
    values = solve_eb(
        cosmologies_of(LCDM, cosmologies), SOLVE_EB_K, SOLVE_EB_A, keep=names
    )
    n_k, n_a = len(SOLVE_EB_K), len(SOLVE_EB_A)
    assert values.shape == (2, len(names), n_k, n_a)

    hist, prepared, k_sorted = _solve_batch(
        LCDM,
        cosmologies_of(LCDM, cosmologies),
        SOLVE_EB_K,
        np.concatenate(([A_START], SOLVE_EB_A)),
    )
    # (n_k, n_c, n_a, nvar), ascending k
    hist = np.asarray(hist).reshape(n_k, 2, -1, prepared.model.nvar)[:, :, 1:, :]
    order = np.argsort(SOLVE_EB_K)
    kept = np.asarray(values)[:, :, order, :]  # ascending k too
    model = (
        prepared.model
    )  # the joint model: the thermal quantities come from the thermal states

    for ci, cosmology in enumerate(cosmologies_of(LCDM, cosmologies)):
        for name in ("delta_c", "delta_b", "F_2", "N_15"):
            j = names.index(name)
            expected = hist[:, ci, :, model.index[name]]
            np.testing.assert_allclose(kept[ci, j], expected, rtol=1e-9, atol=1e-14)

        # The conformal time is one of the states the history holds.
        row, _ = packed(LCDM, cosmology, k_sorted)
        columns = parameter_layout(LCDM).columns
        base = {name: row[:, col][:, None] for name, col in columns.items()}
        base.update(model.state_values(hist[:, ci]))
        base.update(a=SOLVE_EB_A[None, :], k=k_sorted[:, None])
        R, alpha, aH = model.evaluate("R", "alpha", "aH", math=np, **base)
        for name, expected in (("R", R), ("alpha", alpha), ("aH", aH)):
            j = names.index(name)
            np.testing.assert_allclose(kept[ci, j], expected, rtol=2e-6, atol=1e-12)

    # the same thermal error weight as solve_eb's launch, so the two solves agree bit for bit
    pk_batch = _matter_power(
        SOLVE_EB_K, cosmologies_of(LCDM, cosmologies), thermo_weight=THERMO_ERROR_WEIGHT
    )
    for ci, cosmology in enumerate(prepared.cosmologies):
        omega_c, omega_b = cosmology["omega_c"], cosmology["omega_b"]
        delta_c = np.asarray(values[ci, names.index("delta_c"), :, -1])
        delta_b = np.asarray(values[ci, names.index("delta_b"), :, -1])
        delta_m = (omega_c * delta_c + omega_b * delta_b) / (omega_c + omega_b)
        k = np.asarray(SOLVE_EB_K)
        pk = (
            2.0
            * np.pi**2
            / k**3
            * cosmology["A_s"]
            * (k / cosmology["k_p"]) ** (cosmology["n_s"] - 1.0)
            * delta_m**2
        )
        np.testing.assert_allclose(pk, pk_batch[ci], rtol=1e-9)


@pytest.mark.gpu
def test_solve_eb_comoving_curvature_is_conserved_outside_the_horizon():
    """``R`` starts at unity and stays there until the mode enters the horizon.

    The adiabatic initial conditions set ``R = 1``; on super-horizon scales it
    is conserved up to ``O((k tau)^2)`` (0.13 (k tau)^2 in radiation
    domination with no neutrinos), and once inside the horizon it decays with
    the potentials. The conformal time is kept beside it, a state of the solve.
    """

    from discoeb.models import LCDM
    from discoeb.solver._solver import solve_eb

    k = np.array([1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
    a = DIAG_A
    values = solve_eb(
        cosmologies_of(LCDM, _perturbed_cosmologies(1)[0]), k, a, keep=DIAG_KEEP
    )
    assert values.shape == (1, len(DIAG_KEEP), len(k), len(a))
    kept = by_name(values, DIAG_KEEP)
    R, tau = kept["R"][0], kept["tau"][0]
    assert np.isfinite(R).all()

    assert abs(R[0, 0] - 1.0) < 1.0e-6  # k tau = 1e-4
    super_horizon = tau * k[:, None] < 0.1
    assert np.all(np.abs(R[super_horizon] - 1.0) < 5.0e-3), np.abs(
        R[super_horizon] - 1.0
    ).max()
    # k = 0.1 Mpc^-1 entered the horizon long before today and R has decayed.
    assert abs(R[-1, -1]) < 0.5


# --- The kernels a run builds.


KERNEL_BUDGET = {"fast": 6, "all": 16}
"""The kernels a run may build: the fast suite's, and every test's.

Fast, all LCDM's: ``P(k)`` plain, with one and with two sensitivity columns,
``DIAG_KEEP`` plain, the history, and ``REIO_KEEP`` of ``LCDM_REIO``. A
kernel is compiled per number of save times too, so ``P(k)`` is always kept
today alone and the other names on ``DIAG_A``. Slow
adds ``P(k)`` with five columns, the fp64-LU history, plain and one-column
``P(k)`` of ``W0WACDM`` and ``LCDM_MNU``, one column of ``LCDM_REIO``, plain
and one column of ``LCDM_OUTSIDE``, and ``QCDM``."""


@pytest.mark.gpu
def test_the_suite_stays_within_its_kernel_budget(request):
    """The run built no more kernels than its budget: a GPU test keeps the shared names.

    The last test of the last file, so it counts every kernel the run built
    (:func:`helpers.count_kernels`); alone, it counts none. A kernel is 35 s
    to several minutes of compilation, and a test that keeps names of its
    own, saves at a number of scale factors of its own, or differentiates
    in a number of variables no other test does, builds one. Keep
    ``PK_KEEP`` today or ``DIAG_KEEP`` on ``DIAG_A`` instead (``helpers.py``),
    or raise the budget and say here what the kernel is for.
    """

    if not count_kernels():
        pytest.skip("modax's kernel builder is not where the counter looks")
    expression = request.config.getoption("markexpr") or ""
    budget = KERNEL_BUDGET["fast" if "not slow" in expression else "all"]
    built = "\n".join(f"  {i + 1}. {test}" for i, test in enumerate(KERNELS_BUILT))
    assert len(KERNELS_BUILT) <= budget, (
        f"{len(KERNELS_BUILT)} kernels built, the budget is {budget}. "
        f"Each was first needed by:\n{built}"
    )
