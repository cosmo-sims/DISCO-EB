"""The joint kernel (``_kernel``): the two blocks, the parameter row, the initial state and the linear solver's layout."""

import numpy as np
import pytest
from helpers import (
    COSMOLOGY,
    LCDM_T,
    LCDM_X,
    LCDM_XY,
    MODELS,
    QCDM,
    cosmologies_of,
    dense_core_pattern,
    diagonal,
    entries,
    nvar,
    structure,
)

from discoeb.components._component import (
    Interaction,
)
from discoeb.components.species import (
    Fluid,
)
from discoeb.models import LCDM
from discoeb.solver._kernel import (
    joint_kernel,
    joint_model,
    make_initial_states,
    thermal_indices,
)

# --- One component listed several times, each copy under a name of its own.


def test_two_copies_share_a_model():
    from discoeb.solver._kernel import parameter_layout

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


# --- The Einstein-Boltzmann component model, on the host.


def test_make_initial_states_uses_the_model():
    from discoeb.models._cosmology import _as_cosmology

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
    from discoeb.solver._kernel import joint_model

    model = joint_model(LCDM)
    y0 = make_initial_states(cosmology, np.array([1e-2, 1e-1]), model, lna_start=-15.0)
    assert y0.shape == (2, 55) and np.isfinite(y0).all()
    np.testing.assert_allclose(y0[:, 0], [-1e-2, -1e-1], rtol=1e-3)  # etak ~ -k
    np.testing.assert_allclose(
        y0[:, 50:], [[-15.0, 1.0, 2.0, 2.7255 * np.exp(15.0), 0.0]] * 2
    )  # the thermal state


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

    assert LCDM_X.parameter_keys()["Omega_x"] == 0.0
    assert "Omega_x" not in LCDM.parameter_keys()
    layout = parameter_layout(LCDM_X)
    # in component order after the shared nine
    own = parameter_layout(LCDM).directed
    assert layout.directed[:9] == own[:9]
    assert sorted(layout.directed) == sorted((*own, "grhox"))
    assert {"grhox_nominal", "dir_grhox"} <= set(layout.columns)
    i = layout.directed.index("grhox")
    assert layout.columns["dir_grhox"] == layout.ix_dir + i
    assert layout.directed_columns[i] == layout.columns["grhox_nominal"]
    # The built-in models' rows have no trace of it.
    assert "grhox_nominal" not in parameter_layout(LCDM).columns


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
    assert background == ("lna", "thermo", "kappa")
    assert model.published() == ("a", "opacity", "cs2")
    assert recombination.n == 5 and recombination.n_outputs == 3
    assert perturbations.n == nvar and perturbations.reads == (recombination,)
    assert model.state_indices(background) == list(thermal_indices(model))
    # the hierarchy alone, as the model's sparsity traces it: 173 entries
    # (165 and the slip's reading of z under the radiation streaming approximation), 11 colours
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
        LCDM, tuple(_perturbed_cosmologies(1)), k_values, return_stats=True
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

    def d_clxc(self, t):
        return 0.0 * t


def test_a_state_that_reads_no_wave_number_joins_the_background():
    model = joint_model(LCDM_T)
    background, perturbations = model.stages()
    assert background == ("t", "lna", "thermo", "kappa")
    assert "t" not in perturbations and "clxc" in perturbations
    # nobody reads it, so the perturbations are handed what LCDM's are
    assert model.published() == ("a", "opacity", "cs2")
    # without the recombination block the scale factor is an input
    assert LCDM_T.build().stages()[0] == ("t",)
    assert LCDM.build().stages()[0] == ()


def test_a_background_state_the_perturbations_read_is_published():
    model = joint_model(LCDM_T.add(Ageing, name="LCDM+t+ageing"))
    assert model.published() == ("t", "a", "opacity", "cs2")


@pytest.mark.slow
@pytest.mark.gpu
def test_the_kernel_carries_the_state_to_today():
    """The age of the universe from the background block, and LCDM's solve beside it."""

    from discoeb import background_history, solve_eb

    k = np.geomspace(1.0e-3, 0.3, 6)
    tau = np.array([10.0, 300.0, 2.0e4])  # the last lies past tau0
    with_time = solve_eb(
        cosmologies_of(LCDM_T, COSMOLOGY), k, tau, keep=("t", "a", "clxc")
    )
    plain = solve_eb(cosmologies_of(LCDM, COSMOLOGY), k, tau, keep=("a", "clxc"))
    a, tau_bg = background_history(COSMOLOGY)
    age = np.sum(0.5 * (a[1:] + a[:-1]) * np.diff(tau_bg))
    t = np.asarray(with_time["t"])[0]
    np.testing.assert_allclose(t[:, -1], age, rtol=1e-4)
    assert 4000.0 < age < 4500.0  # 13.8 Gyr in Mpc
    np.testing.assert_allclose(
        np.asarray(with_time["clxc"]), np.asarray(plain["clxc"]), rtol=1e-3
    )


# --- A scalar field from outside the package: background states, and a budget closed by shooting.


def test_the_field_is_background_and_its_perturbation_is_not():
    model = joint_model(QCDM)
    background, perturbations = model.stages()
    assert background == ("phi", "phidot", "lna", "thermo", "kappa")
    assert {"dphi", "ddphi"} <= set(perturbations)
    # the perturbations read the field's velocity itself, and the potential
    # stands for the field: its slope and curvature follow from it
    assert model.published() == ("phidot", "V", "a", "opacity", "cs2")
