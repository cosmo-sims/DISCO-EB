"""The joint system's pattern splits into the recombination block and the hierarchy."""

import pytest

from discoeb.perturbations import PerturbationLayout, joint_kernel, thermal_indices


def test_the_recombination_states_are_the_driver_block():
    split_blocks = pytest.importorskip("modax.rodas5P").split_blocks
    colour_sparsity = pytest.importorskip("modax._sparsity").colour_sparsity
    normalize_sparsity = pytest.importorskip("modax._sparsity").normalize_sparsity
    restrict_pattern = pytest.importorskip("modax._split").restrict_pattern

    layout = PerturbationLayout()
    model, _, sparsity = joint_kernel(layout)
    assert model.nvar == layout.nvar + 5
    blocks = split_blocks(sparsity, model.nvar)
    assert blocks.driver == thermal_indices(model)
    assert blocks.n_main == layout.nvar
    main = restrict_pattern(normalize_sparsity(sparsity, model.nvar), blocks.main)
    # the hierarchy alone, as perturbation_sparsity traces it: 165 entries, 11 colours
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
    assert np.all(driver == driver[0]), driver  # the thermal history does not know the wave number
    assert 50 < driver[0] < 400, driver[0]
    assert np.all(driver < accepted[1:].min() * 2), (driver, accepted)  # and the hierarchy steps more at high k
