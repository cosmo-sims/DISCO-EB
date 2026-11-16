"""``W0WACDM`` (``w0wacdm``) over a sweep of parameter space, against CLASS and a converged solve."""

import pytest
from sweep import (
    CROSSING_COSMOLOGY,
    check_dark_energy_crossing_minus_one,
    check_self_convergence,
    check_sweep_is_stable,
    check_sweep_matches_class,
)

SWEEPS = ("w0wacdm",)
"""The sweeps of this module's models, by the names ``sweep.py`` gives them."""


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_is_stable(name):
    check_sweep_is_stable(name)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_matches_class(name):
    check_sweep_matches_class(name)


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "W0WACDM's fluid divides by 1 + w: a w(a) crossing -1 runs out of steps "
        "(20000) and returns a finite, positive P(k) 5.5 to 99 percent off CLASS, "
        f"with no error raised. E.g. {CROSSING_COSMOLOGY}"
    ),
)
@pytest.mark.slow
def test_dark_energy_crossing_minus_one():
    check_dark_energy_crossing_minus_one()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("name", SWEEPS)
def test_self_convergence(name):
    check_self_convergence(name)
