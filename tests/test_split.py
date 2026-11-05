"""The joint system as a modax hierarchy: the recombination block publishing what the hierarchy reads."""

import pytest

from discoeb.perturbations import THERMAL_QUANTITIES, THERMAL_STATES, PerturbationLayout, joint_kernel, thermal_indices


def test_the_recombination_block_publishes_the_thermal_quantities():
    colour_sparsity = pytest.importorskip("modax._sparsity").colour_sparsity

    layout = PerturbationLayout()
    model, hierarchy = joint_kernel(layout)
    assert model.nvar == layout.nvar + 5
    perturbations, recombination = hierarchy.blocks[-1], hierarchy.blocks[0]
    assert hierarchy.layout == (perturbations, recombination)  # the thermal states stay last
    assert recombination.n == 5 and recombination.n_outputs == len(THERMAL_QUANTITIES)
    assert perturbations.n == layout.nvar and perturbations.reads == (recombination,)
    assert model.state_indices(THERMAL_STATES) == list(thermal_indices(model))
    # the hierarchy alone, as perturbation_sparsity traces it: 165 entries, 11 colours
    main = perturbations.sparsity  # normalised by the block: the sorted columns of each row
    assert sum(len(cols) for cols in main) == 165
    assert colour_sparsity(main).n_colours == 11


def test_the_recombination_block_steps_on_its_own():
    """The recombination block takes the same, fewer steps for every wave mode of a cosmology."""
    import numpy as np

    from discoeb.perturbations import (
        PERTURB_ATOL, PERTURB_FIRST_STEP, PERTURB_MAX_STEPS, PERTURB_RTOL, _launch, _pack_batch, _prepare_solve,
    )
    from test_perturbations import _perturbed_cosmologies

    from test_reionization import _cuda_available

    if not _cuda_available():
        pytest.skip("numba-CUDA requires a CUDA GPU")
    prepared = _prepare_solve(tuple(_perturbed_cosmologies(1)))
    k_values = np.array([1.0e-3, 1.0e-2, 1.0e-1, 1.0])
    params, y0, t_span, _, _ = _pack_batch(prepared, k_values)
    _, stats = _launch(
        prepared, params, y0, t_span, rtol=PERTURB_RTOL, atol=PERTURB_ATOL, first_step=PERTURB_FIRST_STEP,
        max_steps=PERTURB_MAX_STEPS, return_stats=True,
    )
    accepted, rejected, loop = (np.asarray(stats[k]) for k in ("accepted_steps", "rejected_steps", "loop_steps"))
    driver = loop - accepted - rejected
    # the thermal history does not know the wave number, but the error the
    # hierarchy inherits from it does, so the block's steps vary with k
    assert 50 < driver.min() and driver.max() < 500, driver
    assert np.all(driver < accepted[1:].min() * 2), (driver, accepted)  # and the hierarchy steps more at high k
