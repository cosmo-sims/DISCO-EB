"""The fused solve (background + recombination + hierarchy in one launch) against the table pipeline."""

import numpy as np
import pytest

from discoeb import perturbations as P
from discoeb.fused import fused_model, solve_matter_power_spectrum_fused_batch
from tests.test_perturbations import MATTER_POWER_K, _perturbed_cosmologies


def test_the_drivers_are_block_lower_triangular():
    """Driver rows hold driver columns only; driver columns are in every row; no fill-in."""
    sparse_direct = pytest.importorskip("modax._sparse_direct")
    normalize = pytest.importorskip("modax._sparsity").normalize_sparsity

    model = fused_model(P.FLAT_MASSLESS_LAYOUT)
    drivers = set(model.drivers())
    assert drivers == {0, 1, 2, 3} and model.nvar == 54
    entries = {(int(r), int(c)) for r, c in model.sparsity()}
    for r, c in entries:
        if r in drivers:
            assert c in drivers
    assert all((r, c) in entries for r in range(model.nvar) for c in drivers)
    assert len(entries) == 238 + 54 * 4
    pattern = normalize(model.sparsity(), model.nvar)
    assert sparse_direct.analyse(pattern, "amd").nnz == len(entries)


@pytest.mark.parametrize("n_cosmologies", [1, 16])
def test_fused_matter_power_matches_the_tables(n_cosmologies):
    pytest.importorskip("numba_cuda_mlir")
    cosmologies = _perturbed_cosmologies(n_cosmologies)
    pk_fused = solve_matter_power_spectrum_fused_batch(MATTER_POWER_K, cosmologies)
    pk_tables = P.solve_matter_power_spectrum_batch(MATTER_POWER_K, cosmologies)
    assert np.abs(pk_fused / pk_tables - 1.0).max() < 2.0e-3
