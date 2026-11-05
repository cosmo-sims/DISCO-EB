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
