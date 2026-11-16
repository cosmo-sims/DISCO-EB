"""``LCDM`` and ``LCDM_MNU`` (``lcdm``) over a sweep of parameter space, against CLASS and a converged solve."""

import pytest
from sweep import (
    check_self_convergence,
    check_sweep_is_stable,
    check_sweep_matches_class,
)

SWEEPS = ("lcdm", "lcdm-mnu")
"""The sweeps of this module's models, by the names ``sweep.py`` gives them."""


@pytest.mark.gpu
@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_is_stable(name):
    check_sweep_is_stable(name)


@pytest.mark.gpu
@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_matches_class(name):
    check_sweep_matches_class(name)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("name", SWEEPS)
def test_self_convergence(name):
    check_self_convergence(name)
