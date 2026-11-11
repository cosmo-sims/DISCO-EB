"""The Jacobian sparsity pattern, and what modax's analysis makes of it.

All host-side: the pattern is traced through the providers of the layout's
component model (:meth:`discoeb.component.Model.sparsity`), and the ordering
and symbolic factorisation that turn it into a compiled solver run on the host
when the kernel is built. Nothing here needs a GPU.
"""

import numpy as np
import pytest
from conftest import HOST, LAYOUTS, ConstantThermalHistory

from discoeb.eb_model import perturbation_model, perturbation_sparsity
from discoeb.perturbations import N_PARAM, PARAM_COLUMNS

sparse_direct = pytest.importorskip("modax._sparse_direct")
colour_sparsity = pytest.importorskip("modax._sparsity").colour_sparsity
normalize_sparsity = pytest.importorskip("modax._sparsity").normalize_sparsity


ORDER_ONE = dict(a=0.5, opacity=3.0, cs2=0.3, rhonu=2.0)
"""Thermal channels of order one: the finite differences below must resolve
every coupling, and at a realistic ``cs2 ~ 1e-8`` or ``a ~ 1e-3`` the baryon
pressure and the dark-energy density fall below the rounding of the rest of
the row."""


def entries(pattern) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in pattern}


def finite_difference_pattern(layout, seed=0, tau=100.0) -> set[tuple[int, int]]:
    """The nonzeros of the host-built right-hand side's Jacobian at a random state and ``tau``."""
    model = perturbation_model(layout, ConstantThermalHistory(**ORDER_ONE))
    rhs = model.build(PARAM_COLUMNS, **HOST)
    rng = np.random.default_rng(seed)
    p = rng.uniform(
        0.5, 1.5, N_PARAM
    )  # order one, curvature and sensitivity direction included
    p[PARAM_COLUMNS["w_DE_0"]] = -0.9
    y = rng.standard_normal(layout.nvar)

    def f(state):
        return np.asarray(rhs(tuple(state), tau, tuple(p)))

    jacobian = np.zeros((layout.nvar, layout.nvar))
    for j in range(layout.nvar):
        e = np.zeros(layout.nvar)
        e[j] = 1e-6
        jacobian[:, j] = (f(y + e) - f(y - e)) / 2e-6
    # A structural zero is exactly zero: the variable never enters the expression.
    return entries(zip(*np.nonzero(np.abs(jacobian) > 1e-9 * np.abs(jacobian).max())))


def structure(layout):
    """``(core, tails)``: the free-streaming tails above the octupole (``E_2`` for the E-modes), the rest core."""
    tails = [
        list(range(layout.ix_g + 3, layout.ix_g + layout.lmaxg + 1)),
        list(range(layout.ix_pol + 1, layout.ix_pol + layout.lmaxpol - 1)),
        list(range(layout.ix_r + 3, layout.ix_r + layout.lmaxr + 1)),
    ]
    if layout.has_massive_neutrinos:
        for q in range(layout.nqmax):
            tails.append([layout.ix_psi(l, q) for l in range(3, layout.lmaxnu + 1)])
    in_tail = {i for tail in tails for i in tail}
    return [i for i in range(layout.nvar) if i not in in_tail], tails


def dense_core_pattern(layout) -> set[tuple[int, int]]:
    """The pattern the states used to declare: a dense core, tridiagonal tails hanging off it."""
    core, tails = structure(layout)
    found = {(r, c) for r in core for c in core}
    for tail in tails:
        parent = tail[0] - 1
        for i, idx in enumerate(tail):
            found.add((idx, idx))
            if i:
                found.add((idx, tail[i - 1]))
            if i + 1 < len(tail):
                found.add((idx, tail[i + 1]))
        found |= {(tail[0], parent), (parent, tail[0])}
    return found


def diagonal(layout) -> set[tuple[int, int]]:
    return {(i, i) for i in range(layout.nvar)}


@pytest.mark.parametrize("name", list(LAYOUTS))
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
    layout = LAYOUTS[name]
    pattern = perturbation_sparsity(layout)

    assert pattern.ndim == 2 and pattern.shape[1] == 2 and pattern.dtype == np.int64
    assert pattern.min() >= 0 and pattern.max() < layout.nvar
    streaming = finite_difference_pattern(layout, tau=1.0)
    approximated = finite_difference_pattern(layout, tau=100.0)
    assert streaming | approximated <= entries(pattern)
    extra = entries(pattern) - (streaming | approximated)
    assert {r for r, _ in extra} <= {layout.ix_vb, layout.ix_g + 1}
    assert approximated < entries(pattern), (
        "the approximation freezes the hierarchies' rows"
    )
    assert not diagonal(layout) <= entries(pattern), (
        "an undamped multipole has no diagonal"
    )


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_the_pattern_has_the_physics_structure(name):
    """A core of metric, fluids and the lowest multipoles, bordered by tridiagonal free-streaming tails.

    This is the structure the states used to declare and the sparse solver
    exploits; the trace finds it in the equations, and finds the core far from
    dense: a free-streaming multipole reads its two neighbours (and itself,
    when Thomson scattering damps it), and the metric reads the densities and
    momenta but not the shears.
    """
    layout = LAYOUTS[name]
    found = entries(perturbation_sparsity(layout))
    core, tails = structure(layout)

    assert found <= dense_core_pattern(layout)
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
    assert {r for r, _ in found} == {c for _, c in found} == set(range(layout.nvar))
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
    layout = LAYOUTS["flat-lcdm"]
    assert layout.nvar == 50
    assert (
        len(perturbation_sparsity(layout))
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


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_amd_fills_in_only_the_core(name):
    """The exact pattern factorises in fewer slots than the dense core ever took.

    Minimum degree still peels each tail from its truncated end, where every
    variable has degree two, so the tails fill in not at all; the core is no
    longer a clique, so eliminating it creates some fill, but never beyond
    the dense core the pattern used to declare -- which factorised with zero
    fill, so its size is exactly what the solver used to work on.
    """
    layout = LAYOUTS[name]
    pattern = normalize_sparsity(perturbation_sparsity(layout), layout.nvar)
    layout_lu = sparse_direct.analyse(pattern, "amd")
    declared = dense_core_pattern(layout)
    core, _ = structure(layout)

    nnz = sum(len(cols) for cols in pattern)
    assert nnz < layout_lu.nnz < len(declared)
    table = layout_lu.slot_table()
    factorised = entries(zip(*np.nonzero(table >= 0)))
    # The factorisation holds every diagonal, for the ``1/(h gamma)``.
    fill = factorised - entries(perturbation_sparsity(layout)) - diagonal(layout)
    assert fill and fill <= {(r, c) for r in core for c in core}
    assert factorised <= declared


def test_the_solver_and_the_kernel_share_one_layout():
    """The slots the sweeps write are the slots the factorisation reads; the fill has its own."""
    layout = LAYOUTS["flat-lcdm"]
    solver = sparse_direct.sparse_direct_solver(
        perturbation_sparsity(layout), layout.nvar
    )
    pattern = normalize_sparsity(perturbation_sparsity(layout), layout.nvar)

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


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_undeclared_diagonals_get_the_factorisations_slots(name):
    """Where ``J`` has no diagonal, the ``1/(h gamma)`` must still reach the factorisation's.

    An undamped multipole's column shares a colour with a neighbour that has
    an entry in its row, so the ``(row, colour)`` grid cell of its diagonal
    belongs to that neighbour. modax used to resolve ``slot(i, i)`` through
    the grid and put the ``1/(h gamma)`` there, for twelve rows of the
    default layout, and the solve returned the initial state; it now goes
    through the factorisation's own diagonal table (modax 9f54d12).
    """
    layout = LAYOUTS[name]
    pattern = normalize_sparsity(perturbation_sparsity(layout), layout.nvar)
    table = sparse_direct.analyse(pattern, "amd").slot_table()
    compressed = sparse_direct.sparse_direct_solver(
        perturbation_sparsity(layout), layout.nvar
    ).compressed

    undeclared = [i for i in range(layout.nvar) if i not in pattern[i]]
    taken = [
        i
        for i in undeclared
        if compressed.packed[i * compressed.n_colours + compressed.colour[i]] >= 0
    ]
    assert taken, "the case the grid lookup got wrong"
    assert all(compressed.slot(i, i) == table[i, i] for i in range(layout.nvar))
    assert compressed.diagonal == tuple(table[i, i] for i in range(layout.nvar))


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_the_exact_pattern_needs_no_more_colours_than_the_dense_core(name):
    layout = LAYOUTS[name]
    exact = colour_sparsity(
        normalize_sparsity(perturbation_sparsity(layout), layout.nvar)
    )
    declared = colour_sparsity(
        normalize_sparsity(np.array(sorted(dense_core_pattern(layout))), layout.nvar)
    )
    assert exact.n_colours < declared.n_colours


def test_the_colouring_keeps_every_entry_apart():
    """Two columns of one colour sharing a row would share a slot.

    modax checks this when it colours; asserting it here is what says the
    *pattern* is the reason it holds, rather than luck in the colouring.
    """
    for name, layout in LAYOUTS.items():
        pattern = normalize_sparsity(perturbation_sparsity(layout), layout.nvar)
        compressed = colour_sparsity(pattern)
        for row, cols in enumerate(pattern):
            colours = {compressed.colour[c] for c in cols}
            assert len(colours) == len(cols), f"{name}: row {row} collides"
