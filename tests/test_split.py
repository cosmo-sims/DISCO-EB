"""The joint system as a modax hierarchy: the recombination block publishing what the hierarchy reads."""

import pytest

from discoeb.kernel import (
    THERMAL_QUANTITIES,
    THERMAL_STATES,
    joint_kernel,
    thermal_indices,
)
from discoeb.models import LCDM


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
    assert recombination.n == 5 and recombination.n_outputs == len(THERMAL_QUANTITIES)
    assert perturbations.n == nvar and perturbations.reads == (recombination,)
    assert model.state_indices(THERMAL_STATES) == list(thermal_indices(model))
    # the hierarchy alone, as the model's sparsity traces it: 173 entries
    # (165 and the slip's reading of z under the radiation streaming approximation), 11 colours
    main = (
        perturbations.sparsity
    )  # normalised by the block: the sorted columns of each row
    assert sum(len(cols) for cols in main) == 173
    assert colour_sparsity(main).n_colours == 11


def test_the_recombination_block_steps_on_its_own():
    """The recombination block takes the same, fewer steps for every wave mode of a cosmology."""
    import numpy as np
    from conftest import _cuda_available
    from test_perturbations import _perturbed_cosmologies

    from discoeb.kernel import _solve_batch

    if not _cuda_available():
        pytest.skip("numba-CUDA requires a CUDA GPU")
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
