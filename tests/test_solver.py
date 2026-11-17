"""Tests of ``discoeb.solver``, one section per module."""

import numpy as np
import pytest
from helpers import (
    COSMOLOGY,
    DENSITY_PARAMETERS,
    LAM,
    LCDM_T,
    LCDM_X,
    LCDM_XY,
    MODELS,
    MORE_PARAMETERS,
    QCDM,
    QCDM_COSMOLOGY,
    SOLVE_EB_K,
    X,
    _matter_power,
    _more_parameter_case,
    _perturbed_cosmologies,
    _split,
    by_name,
    cosmologies_of,
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
    contributes,
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
    # the reionization redshift, declared by the recombination block, comes last
    assert layout.directed[-3:] == ("x_omega", "y_omega", "z_re")
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
    # in component order after the shared seven
    own = parameter_layout(LCDM).directed
    assert layout.directed[:7] == own[:7]
    assert sorted(layout.directed) == sorted((*own, "omega_x"))
    i = layout.directed.index("omega_x")
    assert layout.directed_columns[i] == layout.columns["omega_x"]
    # the directed parameters lead the row, and nothing else stands for them
    assert layout.directed_columns == tuple(range(len(layout.directed)))
    assert not {"eps", "omega_x_nominal", "dir_omega_x"} & set(layout.columns)
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
@pytest.mark.slow
def test_the_recombination_block_steps_on_its_own():
    """The recombination block takes the same, fewer steps for every wave mode of a cosmology."""
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


@pytest.mark.slow
@pytest.mark.gpu
def test_the_kernel_carries_the_state_to_today():
    """The age of the universe from the background block, and LCDM's solve beside it."""

    from discoeb import solve_eb

    k = np.geomspace(1.0e-3, 0.3, 6)
    a = np.array([1.0e-4, 1.0e-2, 1.0])
    with_time = solve_eb(
        cosmologies_of(LCDM_T, COSMOLOGY), k, a, keep=("t", "tau", "delta_c")
    )
    plain = solve_eb(cosmologies_of(LCDM, COSMOLOGY), k, a, keep=("tau", "delta_c"))
    # t = int a dtau = int a / aH dln a, by the trapezoid rule on the host
    grid = np.geomspace(1.0e-9, 1.0, 20000)
    rate = np.asarray(on_the_host(LCDM, COSMOLOGY, "aH", a=grid))
    age = np.trapezoid(grid / rate, np.log(grid))
    conformal_age = np.trapezoid(1.0 / rate, np.log(grid))
    t = np.asarray(with_time[0, 0])  # keep order: t, tau, delta_c
    np.testing.assert_allclose(t[:, -1], age, rtol=1e-4)
    assert 4000.0 < age < 4500.0  # 13.8 Gyr in Mpc
    # and the conformal time, a state now, ends at the conformal age
    np.testing.assert_allclose(with_time[0, 1, :, -1], conformal_age, rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(with_time[:, 2]), np.asarray(plain[:, 1]), rtol=1e-3
    )


# --- A scalar field from outside the package: background states, and a budget closed by shooting.


def test_the_field_is_background_and_its_perturbation_is_not():
    model = joint_model(QCDM)
    background, perturbations = model.stages()
    assert background == ("phi", "phidot", "tau", "thermo", "kappa")
    assert {"dphi", "ddphi"} <= set(perturbations)
    # the perturbations read the field's velocity itself, and the potential
    # stands for the field: its slope and curvature follow from it
    assert model.published() == ("phidot", "tau", "V", "opacity", "cs2")


# --- The columns of a batch, and the keys the solve holds fixed.


def test_a_held_key_raises_when_differentiated():
    """``Y_He`` is read by the solve but not differentiated: no silent zero."""
    import jax

    from discoeb.models._model import _batch_of
    from discoeb.solver._kernel import _columns

    reference = cosmologies_of(LCDM, COSMOLOGY)

    def column(key):
        def of(value):
            model, dicts = _batch_of(reference.replace(**{key: value}))
            return _columns(dicts, model)[key][0]

        return jax.grad(of)(getattr(reference, key))

    assert float(column("omega_c")) == 1.0
    assert float(column("A_s")) == 0.0  # never read by the solve: no tangent
    with pytest.raises(ValueError, match="Y_He"):
        column("Y_He")


# ============================================================================
# sensitivities: modax's forward sensitivities: derivatives of ``solve_eb`` values against central differences
# ============================================================================

# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


@pytest.mark.slow
@pytest.mark.gpu
def test_solve_eb_derivatives():
    """The kept quantities are differentiable in the cosmological parameters.

    The derivative chains the densities' dependence on ``Omega_m``, ``Omega_b``,
    ``H_0``, ``T_cmb`` and ``N_eff`` into the directed inputs, along which modax
    integrates sensitivity columns and stores each quantity's derivative:
    ``jax.jvp``, ``jax.jacfwd`` and ``jax.grad`` of the kept values all come
    from the joint launch. The densities-level derivative is checked
    against central differences of its own primal at fixed thermal history;
    the differences are the noisy side: the solve runs at rtol = atol = 1e-4,
    a step-acceptance flip between the two shifted solves moves a value by
    ~1e-6 of its scale (5e-5 for the photon quadrupole, whose early-time
    values are tiny), so the Jacobian is checked as the change it predicts
    over a 0.1 percent step against the measured change, in units of each
    quantity's scale, and the gradient of a scale-weighted sum over a 1
    percent step. The parameter-level derivative is then the chain rule
    through ``density_parameters``.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import LCDM
    from discoeb.models._model import _batch_of
    from discoeb.solver._kernel import _columns, parameter_layout
    from discoeb.solver._solver import _directed_inputs, solve_eb

    DIFFERENTIABLE_PARAMETERS = DENSITY_PARAMETERS
    cosmology = _perturbed_cosmologies(1)[0]
    # the inputs exactly as solve_eb forms them: a last-bit difference moves a
    # step-acceptance and with it a value by ~1e-6
    theta = jnp.asarray([cosmology[key] for key in DIFFERENTIABLE_PARAMETERS])
    fixed = {k: v for k, v in cosmology.items() if k not in DIFFERENTIABLE_PARAMETERS}

    def inputs_of(theta):
        param = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta))}
        model, dicts = _batch_of(LCDM.from_dict(physical(LCDM, param)))
        columns = _columns(dicts, model)
        return jnp.stack(_directed_inputs(columns, LCDM), axis=1)

    densities = inputs_of(theta)
    a = np.geomspace(1.0e-5, 1.0, 12)
    names = ("R", "delta_c", "F_2")
    noise = {"R": 5.0e-6, "delta_c": 5.0e-6, "F_2": 1.0e-4}

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

    # The joint launch's primal: the fp64 LU in place of the plain fp32 one,
    # so the same trajectories to within the solver's tolerance.
    primal, _ = jax.jvp(kept, (densities,), (jnp.zeros_like(densities),))
    assert np.abs(np.asarray(primal) - base).max() < 1.0e-4 * scale.max()

    # The central differences are converged (rtol = atol = 1e-6): at the default
    # tolerance the solve's own error, different on the two sides of a step,
    # is the larger part of a difference.
    def kept_converged(densities):
        return kept(densities, rtol=1.0e-6, atol=1.0e-6, max_steps=200_000)

    jac = np.asarray(jax.jacfwd(kept)(densities))[..., 0, :]
    DIRECTED_NAMES = parameter_layout(LCDM).directed
    assert jac.shape == base.shape + (len(DIRECTED_NAMES),)
    assert np.isfinite(jac).all()
    for j, dn in enumerate(DIRECTED_NAMES):
        if densities[0, j] == 0.0:  # omega_k, amnu and z_re of a flat LCDM
            continue
        step = 1.0e-3 * float(densities[0, j])
        unit = jnp.zeros_like(densities).at[0, j].set(1.0)
        finite = (
            np.asarray(kept_converged(densities + step * unit))
            - np.asarray(kept_converged(densities - step * unit))
        ) / (2.0 * step)
        err = np.abs(jac[..., j] - finite) * step / scale
        for q, name in enumerate(names):
            # The photon quadrupole only while the hierarchy carries it. Once
            # the radiation streaming approximation is on, its state is
            # frozen where the switch found it, and the switch tests k tau:
            # at fixed scale factor a density moves the conformal time and
            # the switch with it, which a difference sees and a sensitivity,
            # blind to the discontinuity, cannot (5e-4 of the scale).
            compared = err[0, q][:, a < 1.0e-3] if name == "F_2" else err[0, q]
            worst = compared.max()
            assert worst < noise[name], (
                f"d{name}/d{dn}: {worst:.3g} of the scale over a 1e-3 step"
            )

    weights = 1.0 / scale

    @contributes("total")
    def total(d):
        return jnp.sum(kept(d) * weights)

    grad = np.asarray(jax.grad(total)(densities))[0]
    # The transposed rule against the untransposed one: jax.grad of a weighted
    # sum is the Jacobian contracted with the weights (the Jacobian's accuracy
    # is the check above; a central difference of the sum is ill-conditioned,
    # its terms cancelling to 1e-3 of their size for T_cmb).
    summed = np.einsum("cqkt,cqktj->j", np.broadcast_to(weights, base.shape), jac)
    np.testing.assert_allclose(
        grad, summed, rtol=1.0e-6, err_msg="grad vs contracted jacfwd"
    )

    # Through the dictionary: the chain rule with the densities' Jacobian,
    # under jacfwd and grad alike, and a plain number where a derivative is
    # not carried.
    def of_parameters(theta):
        param = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta))}
        return solve_eb(cosmologies_of(LCDM, param), SOLVE_EB_K, a, keep=names)

    jac_theta = np.asarray(jax.jacfwd(of_parameters)(theta))
    d_densities = np.asarray(jax.jacfwd(lambda t: inputs_of(t)[0])(theta))
    # The two are separate launches, one replica per parameter along
    # d inputs / d theta against one per input, so they differ by the
    # solve's noise: checked like the central differences above, as the
    # change each predicts over a 0.1 percent step in units of the
    # quantity's scale.
    chained = np.einsum("cqktj,jp->cqktp", jac, d_densities)
    for p, key in enumerate(DIFFERENTIABLE_PARAMETERS):
        step = 1.0e-3 * float(theta[p])
        err = np.abs(jac_theta[..., p] - chained[..., p]) * step / scale
        for q, name in enumerate(names):
            assert err[0, q].max() < noise[name], (
                f"d{name}/d{key}: {err[0, q].max():.3g} against the chain rule"
            )
    grad_theta = np.asarray(
        jax.grad(lambda t: jnp.sum(of_parameters(t) * weights))(theta)
    )
    np.testing.assert_allclose(
        grad_theta, np.einsum("cqkt,cqktp->p", weights, jac_theta), rtol=1e-10
    )
    assert np.all(grad_theta != 0.0)

    # m_nu needs MassiveNeutrinos, cs2_DE DarkEnergy. The sensitivities in
    # Y_He and cs2_DE miss central differences of P(k) by 0.8 and 1 percent
    # (2026-11-13), so they stay held.
    # z_re needs ReionizationDepth, whose state resolves its derivative.
    for key, value in (("m_nu", 0.0), ("Y_He", 0.245), ("z_re", 7.0)):
        with pytest.raises(ValueError, match="cannot be differentiated"):
            jax.grad(
                lambda v, key=key: jnp.sum(
                    solve_eb(
                        cosmologies_of(LCDM, {**cosmology, key: v}),
                        SOLVE_EB_K,
                        a,
                        keep=names,
                    )
                )
            )(value)


# Measured |sensitivity - central difference| / max |central difference|
# (2026-11-13, RTX 4070 SUPER), largest over R and delta_b: w_DE_0 1.6e-4,
# w_DE_a 3.4e-4, Omega_k 2.7e-4, m_nu 9.8e-4 (dR/dmnu). Of sum P(k): w_DE_0
# 6.9e-5, w_DE_a 1.3e-6, Omega_k 1.5e-5, m_nu 7.2e-5. The gate is 1e-3, 2e-3
# for m_nu's solve. Y_He and cs2_DE missed P(k) by 0.8 and 1 percent at every
# step tried and stay held.
MORE_SOLVE_GATES = {"m_nu": 2.0e-3}


MORE_NAMES = ("R", "delta_b", "kappa")


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("name", list(MORE_PARAMETERS))
def test_more_parameters_solve_eb(name):
    """``solve_eb``'s ``jax.jvp`` in each parameter beyond the densities' against central differences.

    The differences come from converged solves (rtol = atol = 1e-6), the
    sensitivity from one at the default tolerance, as in
    :func:`test_solve_eb_derivatives`. ``kappa``, the optical depth, carries
    the thermal parameters and the reionization redshift directly.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.solver._solver import solve_eb

    model, cosmology, step = _more_parameter_case(name)
    a = np.geomspace(1.0e-5, 1.0, 12)

    def values(p, **options):
        batch = cosmologies_of(model, {**cosmology, name: p})
        return solve_eb(batch, SOLVE_EB_K, a, keep=MORE_NAMES, **options)

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
    for q, quantity in enumerate(MORE_NAMES):
        scale = np.abs(value[0, q]).max()
        err = np.abs(tangent[0, q] - finite[0, q]).max()
        signal = np.abs(finite[0, q]).max()
        print(
            f"d{quantity}/d{name}: {err / signal:.3g} of the largest difference, "
            f"which moves the quantity by {signal * step / scale:.3g} of its scale"
        )
        if signal * step < 1.0e-4 * scale:
            # A change the solve's noise swamps: the sensitivity must predict
            # one as small.
            assert err * step < 1.0e-5 * scale, f"d{quantity}/d{name}"
            continue
        assert err < MORE_SOLVE_GATES.get(name, 1.0e-3) * signal, (
            f"d{quantity}/d{name}: {err / signal:.3g} of its largest central difference"
        )


# ============================================================================
# solver: ``solve_eb`` (``_solver``): kept quantities, and solves of models with components added in the tests
# ============================================================================

# --- One component listed several times, each copy under a name of its own.


OMEGA_FLUID = 0.08


def test_a_dictionary_names_no_model():
    """No model stands in for a bare dictionary: the caller names one."""

    from discoeb import directed_inputs

    with pytest.raises(TypeError, match="LCDM"):
        directed_inputs(COSMOLOGY)
    directed_inputs(cosmologies_of(LCDM, COSMOLOGY))


@pytest.mark.slow
@pytest.mark.gpu
def test_splitting_the_dark_matter_into_a_named_fluid_leaves_the_solve_unchanged():
    from discoeb import solve_eb

    model = LCDM.add(X, name="LCDM+x")
    k = np.geomspace(1.0e-3, 0.3, 8)
    a = np.geomspace(1.0e-6, 1.0, 16)
    lcdm = cosmologies_of(LCDM, COSMOLOGY)
    with_fluid = cosmologies_of(
        model,
        {
            **COSMOLOGY,
            "Omega_m": COSMOLOGY["Omega_m"] - OMEGA_FLUID,
            "x_Omega": OMEGA_FLUID,
        },
    )
    names = ("delta_c", "R", "tau")
    plain = by_name(solve_eb(lcdm, k, a, keep=names), names)
    keep = (*names, "x_delta")
    split = by_name(solve_eb(with_fluid, k, a, keep=keep), keep)
    # the same expansion: the conformal time, a state, comes out the same
    np.testing.assert_allclose(split["tau"], plain["tau"], rtol=1e-5)
    np.testing.assert_allclose(split["x_delta"], split["delta_c"], rtol=1e-5)
    for name in ("delta_c", "R"):
        ours, theirs = split[name], plain[name]
        assert np.abs(ours - theirs).max() < 1e-3 * np.abs(theirs).max(), name


# --- A species with a parameter of its own, added without touching the package.


SOLVE_K = np.geomspace(1.0e-3, 0.3, 8)


SOLVE_A = np.geomspace(1.0e-6, 1.0, 16)


@pytest.mark.slow
@pytest.mark.gpu
def test_splitting_the_dark_matter_leaves_the_solve_unchanged():
    """Two pressureless fluids holding the CDM between them evolve as the CDM did.

    The second fluid's contrast equals the CDM's in the same solve, the
    contrast and the curvature perturbation agree with the plain LCDM solve
    to the solver's tolerance, and so does the conformal time. The directed
    input ``omega_x`` moves every kept quantity as ``omega_c`` does.
    """

    import jax
    import jax.numpy as jnp

    from discoeb import directed_inputs, solve_eb

    names = ("delta_c", "R", "F_2", "tau")
    lcdm = cosmologies_of(LCDM, COSMOLOGY)
    plain = by_name(solve_eb(lcdm, SOLVE_K, SOLVE_A, keep=names), names)
    batch = cosmologies_of(LCDM_X, _split(COSMOLOGY))
    keep = (*names, "delta_x")
    split = by_name(solve_eb(batch, SOLVE_K, SOLVE_A, keep=keep), keep)

    np.testing.assert_allclose(split["tau"], plain["tau"], rtol=1e-5)
    names = names[:-1]
    # The same equations and initial state; the fp32 LU tells the columns apart.
    np.testing.assert_allclose(split["delta_x"], split["delta_c"], rtol=1e-5)
    # The initial states agree (the growing mode's om sums rho_matter, which
    # the second fluid contributes to), so what is left is the solver's:
    # the smooth quantities to 1e-3, the oscillating photon quadrupole,
    # stepped differently by a system one state larger, to a few times that.
    for name, gate in (("delta_c", 1e-3), ("R", 1e-3), ("F_2", 5e-3)):
        a, b = split[name], plain[name]
        assert np.abs(a - b).max() < gate * np.abs(b).max(), name

    from discoeb.solver._kernel import parameter_layout

    directed = parameter_layout(LCDM_X).directed
    x = directed_inputs(batch)
    assert x.shape == (1, len(directed))

    def kept(inputs):
        return solve_eb(batch, SOLVE_K, SOLVE_A, keep=names, inputs=inputs)

    jac = np.asarray(jax.jacfwd(kept)(jnp.asarray(x)))[..., 0, :]
    assert np.isfinite(jac).all()
    along_c = jac[..., directed.index("omega_c")]
    along_x = jac[..., directed.index("omega_x")]
    scale = np.abs(along_c).max(axis=(-2, -1), keepdims=True)
    assert np.abs(along_x - along_c).max() < 1e-3 * scale.max()


# --- A scalar field from outside the package: background states, and no budget to close.


def _power(model, cosmology, k):
    from discoeb import matter_power_spectrum, solve_eb

    cosmology = cosmologies_of(model, cosmology)
    keep = ("delta_c", "delta_b")
    values = solve_eb(cosmology, k, [1.0], keep=keep, thermo_weight=0.1)
    today = {n: v[0, :, -1] for n, v in by_name(values, keep).items()}
    names = ("omega_c", "omega_b", "A_s", "n_s", "k_p")
    return np.asarray(
        matter_power_spectrum(k, **today, **{n: getattr(cosmology, n) for n in names})
    )


GROWTH_A = np.geomspace(1.0e-5, 1.0, 2000)


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
    rate = np.asarray(solve_eb(cosmology, [1.0e-2], a, keep=("aH",)))[0, 0, 0]
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

    kept = np.asarray(
        solve_eb(cosmologies_of(QCDM, rolling), k, [1.0], keep=("phi", "phidot"))
    )[0, :, :, -1]
    np.testing.assert_allclose(kept, kept[:, :1] * np.ones(len(k)), rtol=1e-6)
    assert np.all(kept > 0.0)
    ratio = field / lcdm
    growth = (growth_field / growth_lcdm) ** 2
    assert abs(growth - 1.0) > 3e-3  # the field's history is not LCDM's
    np.testing.assert_allclose(ratio[k > 0.01], growth, rtol=1e-3)
    # near the horizon the field's own perturbation takes part
    assert abs(ratio[0] - ratio[-1]) > 3e-3


# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


SOLVE_EB_A = np.geomspace(1.0e-6, 1.0, 48)


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
    names = ("R", "delta_c", "delta_b", "F_2", "N_15", "alpha", "aH")
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


@pytest.mark.slow
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
    a = np.geomspace(2.0e-6, 1.0, 64)
    values = solve_eb(
        cosmologies_of(LCDM, _perturbed_cosmologies(1)[0]), k, a, keep=("R", "tau")
    )
    assert values.shape == (1, 2, len(k), len(a))
    R, tau = np.asarray(values[0, 0]), np.asarray(values[0, 1])
    assert np.isfinite(R).all()

    assert abs(R[0, 0] - 1.0) < 1.0e-6  # k tau = 1e-4
    super_horizon = tau * k[:, None] < 0.1
    assert np.all(np.abs(R[super_horizon] - 1.0) < 5.0e-3), np.abs(
        R[super_horizon] - 1.0
    ).max()
    # k = 0.1 Mpc^-1 entered the horizon long before today and R has decayed.
    assert abs(R[-1, -1]) < 0.5
