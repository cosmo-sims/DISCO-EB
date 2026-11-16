"""A scalar field from outside the package: background states, and a budget closed by shooting.

``Quintessence`` is a minimally coupled field ``phi`` (in reduced Planck
units) with the potential ``V = V0 exp(-lam phi)``. Its homogeneous value
and velocity are background states, which no built-in model has: the trace
finds them, the host integrates them, and ``V0`` closes the budget as the
root of ``H(a = 1) = H0``. With ``lam = 0`` the field is a cosmological
constant, so the model must reproduce LCDM.
"""

import math

import numpy as np
import pytest
from conftest import _cuda_available, cosmologies_of
from jax.typing import ArrayLike

from discoeb.components._component import Derived, Parameter, Species, State
from discoeb.components.species import CosmologicalConstant
from discoeb.models import LCDM
from discoeb.solver._kernel import joint_model


class Quintessence(Species):
    """``phi'' + 2 aH phi' + a^2 V_phi = 0`` and its synchronous-gauge perturbation.

    Densities are ``8 pi G rho a^2 = phi'^2 / 2 + a^2 V`` with ``V`` in
    1/Mpc^2. The perturbation obeys ``dphi'' + 2 aH dphi' + (k^2 + a^2
    V_phiphi) dphi = -k z phi'``. The field starts frozen.
    """

    phi = State()
    phidot = State()
    dphi = State()
    ddphi = State()

    lam: ArrayLike = Parameter(0.0, column=True)
    V0: ArrayLike = Derived(directed=True, closes_budget=True)

    def V(self, V0, lam, phi):
        return V0 * math.exp(-lam * phi)

    def V_phi(self, V, lam):
        return -lam * V

    def V_phiphi(self, V, lam):
        return lam * lam * V

    def grho(self, phidot, V, a2):
        return 0.5 * phidot * phidot + a2 * V

    def gpres(self, phidot, V, a2):
        return 0.5 * phidot * phidot - a2 * V

    def dgrho(self, phidot, ddphi, a2, V_phi, dphi):
        return phidot * ddphi + a2 * V_phi * dphi

    def dgq(self, k, phidot, dphi):
        return k * phidot * dphi

    def d_phi(self, phidot):
        return phidot

    def d_phidot(self, phidot, adotoa, a2, V_phi):
        return -2.0 * adotoa * phidot - a2 * V_phi

    def d_dphi(self, ddphi):
        return ddphi

    def d_ddphi(self, ddphi, dphi, adotoa, k, a2, V_phiphi, z, phidot):
        return -2.0 * adotoa * ddphi - (k * k + a2 * V_phiphi) * dphi - k * z * phidot


QCDM = LCDM.swap(CosmologicalConstant, Quintessence, name="QCDM")

COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "k_p": 0.05,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
}
LAM = 1.0


def _densities(model, cosmology):
    from discoeb.models._cosmology import _columns
    from discoeb.solver._background import density_parameters

    columns = _columns([cosmology], held=False, keys=model.parameter_keys())
    return {k: v[0] for k, v in density_parameters(columns, model=model).items()}


def test_the_field_is_background_and_its_perturbation_is_not():
    model = joint_model(QCDM)
    background, perturbations = model.stages()
    assert background == ("phi", "phidot", "lna", "thermo", "kappa")
    assert {"dphi", "ddphi"} <= set(perturbations)
    # the perturbations read the field's velocity itself, and the potential
    # stands for the field: its slope and curvature follow from it
    assert model.published() == ("phidot", "V", "a", "opacity", "cs2")


def test_a_flat_potential_closes_the_budget_as_a_cosmological_constant():
    lcdm = _densities(LCDM, COSMOLOGY)
    flat = _densities(QCDM, COSMOLOGY)
    assert float(flat["V0"]) == pytest.approx(float(lcdm["grhov"]), rel=1e-9)


def test_shooting_closes_the_budget_of_a_rolling_field():
    """``V0`` is the root of ``H(a = 1) = H0`` through the field's integration."""

    import jax
    import jax.numpy as jnp

    from discoeb.solver._background import _background_solution, background_model

    d = _densities(QCDM, {**COSMOLOGY, "lam": LAM})
    a, tau, states = _background_solution(d, QCDM)
    today = {name: values[-1] for name, values in states.items()}
    adotoa, grho, gpres = background_model(QCDM).evaluate(
        "adotoa", "grho", "gpres", math=jnp, a=1.0, **today, **d
    )
    assert float(3.0 * adotoa**2 / d["grhom"]) == pytest.approx(1.0, abs=1e-10)
    # the field has rolled: more potential than LCDM's constant to start with,
    # and an equation of state above -1 today
    assert float(d["V0"]) > float(_densities(LCDM, COSMOLOGY)["grhov"])
    kinetic = 0.5 * float(today["phidot"]) ** 2
    potential = float(d["V0"]) * math.exp(-LAM * float(today["phi"]))
    w0 = (kinetic - potential) / (kinetic + potential)
    assert -0.95 < w0 < -0.7
    assert float(today["phi"]) > 0.0

    # and the closure is differentiable: more matter leaves less for the field
    def V0_of(Omegam):
        return _densities(QCDM, {**COSMOLOGY, "lam": LAM, "Omegam": Omegam})["V0"]

    Omegam = COSMOLOGY["Omegam"]
    slope = float(jax.jacfwd(V0_of)(jnp.asarray(Omegam)))
    h = 1e-4
    central = float(V0_of(Omegam + h) - V0_of(Omegam - h)) / (2 * h)
    assert slope < 0.0
    assert slope == pytest.approx(central, rel=1e-5)


def _power(model, cosmology, k):
    from discoeb import conformal_ages, matter_power_spectrum, solve_eb

    batch = cosmologies_of(model, [cosmology])
    tau0 = conformal_ages(batch)[0]
    solution = solve_eb(
        batch, k, tau0[None], keep=("clxc", "clxb", "z", "vb", "a"), thermo_weight=0.1
    )
    today = {n: solution[n][0, :, -1] for n in ("clxc", "clxb", "z", "vb")}
    pk = matter_power_spectrum(
        k,
        **today,
        tau0=tau0,
        **{n: cosmology[n] for n in ("Omegam", "Omegab", "A_s", "n_s", "k_p")},
    )
    return np.asarray(pk), np.asarray(solution["a"])[0, :, -1]


def _growth(model, cosmology):
    """The matter growth factor today in the model's background: ``delta'' + aH delta' = 4 pi G a^2 rho_m delta``.

    Heun steps in ``ln a`` on the background's own grid, from ``a = 1e-5``.
    Only the ratio between two models that agree early on is meaningful.
    """

    import jax.numpy as jnp

    from discoeb.solver._background import _background_solution, background_model

    d = _densities(model, cosmology)
    a, _, states = _background_solution(d, model)
    rate = np.asarray(
        background_model(model).evaluate("adotoa", math=jnp, a=a, **states, **d)
    )
    a = np.asarray(a)
    source = 0.5 * float(d["grhoc"] + d["grhob"]) / a
    lna = np.log(a)
    delta, slope = 1.0, 0.0
    for i in range(int(np.searchsorted(a, 1.0e-5)), len(a) - 1):
        h = lna[i + 1] - lna[i]
        f0 = (slope / rate[i], (-rate[i] * slope + source[i] * delta) / rate[i])
        d1, s1 = delta + h * f0[0], slope + h * f0[1]
        f1 = (
            s1 / rate[i + 1],
            (-rate[i + 1] * s1 + source[i + 1] * d1) / rate[i + 1],
        )
        delta += 0.5 * h * (f0[0] + f1[0])
        slope += 0.5 * h * (f0[1] + f1[1])
    return delta


@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")
def test_the_solve_matches_lcdm_and_the_growth_in_the_fields_background():
    """A flat potential gives LCDM, and a rolling field suppresses ``P(k)`` as its expansion history says.

    Inside the horizon a field with unit sound speed does not cluster, so
    the power relative to LCDM is the squared ratio of the growth factors
    in the two backgrounds, which the host integrates on its own. (A CPL
    fluid through the field's equation of state would cross ``w = -1`` in
    the past, which the fluid equations cannot.)
    """

    k = np.geomspace(1.0e-3, 0.3, 8)
    lcdm, _ = _power(LCDM, COSMOLOGY, k)
    flat, a_flat = _power(QCDM, COSMOLOGY, k)
    np.testing.assert_allclose(a_flat, 1.0, rtol=1e-4)
    np.testing.assert_allclose(flat, lcdm, rtol=2e-3)

    rolling = {**COSMOLOGY, "lam": LAM}
    field, a_field = _power(QCDM, rolling, k)
    # the kernel's own integration of the field ends today where the host's did
    np.testing.assert_allclose(a_field, 1.0, rtol=1e-4)
    suppression = field / lcdm
    growth = (_growth(QCDM, rolling) / _growth(LCDM, COSMOLOGY)) ** 2
    assert 0.9 < growth < 0.99  # dark energy that sets in earlier: less growth
    np.testing.assert_allclose(suppression[k > 0.01], growth, rtol=1e-3)
    # near the horizon the field's own perturbation takes part
    assert suppression[0] > suppression[-1] + 3e-3
