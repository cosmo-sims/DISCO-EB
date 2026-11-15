"""A background state added from outside: the equations split the blocks themselves.

``CosmicTime`` integrates ``dt = a dtau``. Its rate reads no wave number, so
:meth:`Equations.stages <discoeb.component.Equations.stages>` puts it in the
background block beside the recombination history, the host integrates it
from the start of the background, and the kernel carries it on from there.
"""

import numpy as np
import pytest
from conftest import _cuda_available, cosmologies_of

from discoeb.component import Component, Interaction, State
from discoeb.kernel import joint_model
from discoeb.models import LCDM


class CosmicTime(Component):
    """The cosmic time ``t`` in Mpc: ``dt / dtau = a``."""

    t = State()

    def d_t(self, a):
        return a


class Ageing(Interaction):
    """A perturbation equation that reads the background state itself."""

    def d_clxc(self, t):
        return 0.0 * t


LCDM_T = LCDM.add(CosmicTime, name="LCDM+t")

COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
}


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


def test_the_host_integrates_the_states_of_the_background():
    """``t = int a dtau``, by Runge-Kutta steps beside the conformal time."""

    import jax.numpy as jnp

    from discoeb.background import (
        _background_solution,
        background_endpoints,
        density_parameters,
    )
    from discoeb.cosmology import _columns

    columns = _columns([COSMOLOGY], held=False, keys=LCDM_T.parameter_keys())
    densities = {k: v[0] for k, v in density_parameters(columns, model=LCDM_T).items()}
    a, tau, states = _background_solution(densities, LCDM_T)
    a_lcdm, tau_lcdm, none = _background_solution(densities, LCDM)
    assert none == {}
    # the conformal time of LCDM, whose trapezoid rule is the less exact (2e-6)
    np.testing.assert_allclose(tau, tau_lcdm, rtol=5e-6)
    t = np.concatenate(([0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(tau_lcdm))))
    np.testing.assert_allclose(states["t"][1:], t[1:], rtol=1e-4)
    tau0, _, _, start = background_endpoints(columns, LCDM_T)
    assert set(start) == {"t"}
    assert float(start["t"][0]) == pytest.approx(
        float(jnp.interp(0.1, tau, states["t"])), rel=1e-12
    )
    assert float(tau0[0]) == pytest.approx(float(tau_lcdm[-1]), rel=5e-6)


@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")
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
