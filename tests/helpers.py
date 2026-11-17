"""
Helpers including host stand-ins, models, components added from outside
and reference cosmologies.

Not a test file. ``conftest.py`` sets the environment before this imports
:mod:`discoeb`; the test folders import from here by name (``pythonpath`` in
``pyproject.toml``).
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from jax.typing import ArrayLike

import discoeb
from discoeb.components._component import (
    Component,
    Parameter,
    Species,
    State,
    contributes,
    derivative,
)
from discoeb.components.gravity import H100_SQ
from discoeb.components.recombination import ReionizationDepth
from discoeb.components.species import (
    CosmologicalConstant,
    Fluid,
    MassiveNeutrinos,
)
from discoeb.models import LCDM, LCDM_MNU, W0WACDM, W0WACDM_MNU
from discoeb.solver._kernel import parameter_layout

WIDEST_LAYOUT = parameter_layout(W0WACDM_MNU)
"""The parameter row of W0WACDM_MNU, whose columns are a superset of every
built-in model's: a host test may build any built-in model on it."""
PARAM_COLUMNS = WIDEST_LAYOUT.columns
N_PARAM = WIDEST_LAYOUT.n


def _cuda_available() -> bool:
    """Whether numba-CUDA sees a GPU, for tests that launch a kernel directly."""

    try:
        from numba_cuda_mlir import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


# Host stand-ins for a build: an identity jit, numpy scratch arrays, float64.
HOST = dict(
    jit=lambda f: f,
    cuda=SimpleNamespace(
        local=SimpleNamespace(array=lambda shape, dtype: np.zeros(shape))
    ),
    float64=np.float64,
)

MODELS = {
    "flat-lcdm": LCDM,
    "dark-energy": W0WACDM,
    "massive-nu": LCDM_MNU.with_settings(NQMAX=3, LMAX_NU=8),
    # Blocks of unequal length: the hand-written Schur solver this replaced
    # could not take them.
    "ragged-blocks": W0WACDM_MNU.with_settings(
        LMAX_G=11, LMAX_POL=13, LMAX_NR=17, NQMAX=2, LMAX_NU=9
    ),
}


def cosmologies_of(model, params):
    """The cosmologies of parameter dictionaries: one for a dictionary, a list for a list.

    Through :meth:`Model.from_dict <discoeb.models.Model.from_dict>`, so a
    dictionary written for another model passes while its other keys keep
    their defaults. One in ``Omega_m``, ``Omega_b`` and ``H_0`` is converted
    first (:meth:`Model.physical_parameters
    <discoeb.models.Model.physical_parameters>`).
    """

    if not isinstance(params, dict):
        return [cosmologies_of(model, p) for p in params]
    if "Omega_m" in params:
        params = model.physical_parameters(params)
    return model.from_dict(params)


def physical(model, params) -> dict:
    """A dictionary in ``Omega_m``, ``Omega_b`` and ``H_0`` in the model's own parameters."""

    return model.physical_parameters(params)


def packed(model, cosmology, k):
    """``(params, y0)`` of one cosmology at the wave modes ``k``, as the kernel is launched with them."""

    from discoeb.models._model import _batch_of
    from discoeb.solver._kernel import _columns, _pack

    model, dicts = _batch_of(cosmology)
    columns = _columns(dicts, model, held=False)
    params, y0 = _pack(model.resolved(), (), tuple(np.asarray(k).tolist()), columns)
    return np.asarray(params), np.asarray(y0)


def model_parameters(model, cosmology) -> dict:
    """Every parameter of the model, derived ones included, for a cosmology or a dictionary of one."""

    from discoeb.solver._kernel import density_parameters

    if isinstance(cosmology, dict):
        cosmology = cosmologies_of(model, cosmology)
    return density_parameters(cosmology.to_dict(), xp=np, model=model)


def on_the_host(model, cosmology, *quantities, **base):
    """Quantities of the model's equations evaluated on the host in JAX, at ``base`` (``a=...``)."""

    import jax.numpy as jnp

    from discoeb.models._model import _host_equations

    values = model_parameters(model, cosmology)
    return _host_equations(model).evaluate(*quantities, math=jnp, **base, **values)


def state_index(model) -> dict[str, int]:
    """The perturbation state's indices by element name."""

    return {name: i for i, name in enumerate(model.state_names())}


class ConstantThermalHistory(Component):
    """A thermal history frozen at one epoch, standing in for the recombination block.

    The defaults are a realistic epoch; a test that differences the
    right-hand side passes channels of order one, since at ``cs2 ~ 1e-8`` or
    ``a ~ 1e-3`` the baryon pressure and the dark-energy density fall below
    the rounding of the rest of the row.
    """

    def __init__(self, a=1e-3, opacity=50.0, cs2=1e-8):
        self.a = lambda: a
        self.opacity = lambda: opacity
        self.cs2 = lambda: cs2


# --- One component listed several times, each copy under a name of its own.


X, Y = Fluid.named("x"), Fluid.named("y")


LCDM_XY = LCDM.add(X, Y, name="LCDM+x+y")


COSMOLOGY = {
    "Omega_m": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omega_b": 0.02238280 / 0.6732117**2,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H_0": 67.32117,
    "T_cmb": 2.7255,
    "Y_He": 0.2454006,
    "N_eff": 3.046,
}


# --- The Einstein-Boltzmann component model, on the host.


def build_rhs(cosmology_model):
    model = cosmology_model.build(ConstantThermalHistory())
    return model, model.build(PARAM_COLUMNS, **HOST)


# --- The Jacobian sparsity pattern, and what modax's analysis makes of it.


def nvar(model) -> int:
    return len(model.state_names())


def entries(pattern) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in pattern}


def structure(model):
    """``(core, tails)``: the free-streaming tails above the octupole (``E_2`` for the E-modes), the rest core."""
    ix = state_index(model)
    tails = [
        [ix[f"F_{ell}"] for ell in range(3, model.setting("LMAX_G") + 1)],
        [ix[f"E_{i}"] for i in range(1, model.setting("LMAX_POL") - 1)],
        [ix[f"N_{ell}"] for ell in range(3, model.setting("LMAX_NR") + 1)],
    ]
    if model.has(MassiveNeutrinos):
        for q in range(model.setting("NQMAX")):
            lmax = model.setting("LMAX_NU")
            tails.append([ix[f"psi_{q}_{ell}"] for ell in range(3, lmax + 1)])
    in_tail = {i for tail in tails for i in tail}
    return [i for i in range(nvar(model)) if i not in in_tail], tails


def dense_core_pattern(model) -> set[tuple[int, int]]:
    """The pattern the states used to declare: a dense core, tridiagonal tails hanging off it."""
    core, tails = structure(model)
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


def diagonal(model) -> set[tuple[int, int]]:
    return {(i, i) for i in range(nvar(model))}


# --- A species with a parameter of its own, added without touching the package.


class SecondDarkMatter(Species):
    """Pressureless and at rest in synchronous gauge, like CDM: ``delta_x' = -h' / 2``."""

    delta_x = State(initial=lambda F: 0.75 * F[0])

    omega_x: ArrayLike = Parameter(0.0, directed=True)

    @classmethod
    def check(cls, omega_x):
        if omega_x < 0.0:
            raise ValueError(f"omega_x = {omega_x} must not be negative")

    def rho_x(self, omega_x, a):
        inv_a = 1.0 / a
        return omega_x * inv_a * inv_a * inv_a

    @contributes("rho")
    def rho(self, rho_x):
        return rho_x

    @contributes("rho_matter")
    def rho_matter(self, rho_x):
        """It is matter to the growing mode's initial conditions."""
        return rho_x

    @contributes("delta_rho")
    def delta_rho(self, rho_x, delta_x):
        return rho_x * delta_x

    @derivative("delta_x")
    def d_delta_x(self, h_prime):
        return -0.5 * h_prime


LCDM_X = LCDM.add(SecondDarkMatter, name="LCDM+x")


LCDM_REIO = LCDM.add(ReionizationDepth, name="LCDM+reio")
"""LCDM with the reionization depth as a state: ``tau_reio`` and the ``z_re`` derivative."""


OMEGA_X = 0.08


def _split(cosmology, omega_x=OMEGA_X):
    """The same universe with ``omega_x`` of its cold dark matter in the second fluid."""

    return {**cosmology, "Omega_m": cosmology["Omega_m"] - omega_x, "Omega_x": omega_x}


# --- A background state added from outside: the equations split the blocks themselves.


class CosmicTime(Component):
    """The cosmic time ``t`` in Mpc: ``dt / dtau = a``."""

    t = State()

    @derivative("t")
    def d_t(self, a):
        return a


LCDM_T = LCDM.add(CosmicTime, name="LCDM+t")


# --- A scalar field from outside the package: background states, and a budget closed by shooting.


class Quintessence(Species):
    """``phi'' + 2 aH phi' + 3 H100^2 a^2 V_phi = 0`` and its synchronous-gauge perturbation.

    The potential ``V`` is a density in units of ``omega``, and the field's
    density is ``phi'^2 / (6 H100^2 a^2) + V``. The perturbation obeys
    ``dphi'' + 2 aH dphi' + (k^2 + 3 H100^2 a^2 V_phiphi) dphi = -h' phi' /
    2``. The field starts frozen.
    """

    phi = State()
    phidot = State()
    dphi = State()
    ddphi = State()

    lam: ArrayLike = Parameter(0.0, column=True)
    V0: ArrayLike = Parameter(directed=True)  # in units of omega, as omega_de

    def V(self, V0, lam, phi):
        return V0 * math.exp(-lam * phi)

    def V_phi(self, V, lam):
        return -lam * V

    def V_phiphi(self, V, lam):
        return lam * lam * V

    def kinetic(self, phidot, a):
        return phidot * phidot / (6.0 * H100_SQ * a * a)

    @contributes("rho")
    def rho(self, kinetic, V):
        return kinetic + V

    @contributes("pressure")
    def pressure(self, kinetic, V):
        return kinetic - V

    @contributes("delta_rho")
    def delta_rho(self, phidot, ddphi, a, V_phi, dphi):
        return phidot * ddphi / (3.0 * H100_SQ * a * a) + V_phi * dphi

    @contributes("momentum")
    def momentum(self, k, phidot, dphi, a):
        return k * k * phidot * dphi / (3.0 * H100_SQ * a * a)

    @derivative("phi")
    def d_phi(self, phidot):
        return phidot

    @derivative("phidot")
    def d_phidot(self, phidot, aH, a, V_phi):
        return -2.0 * aH * phidot - 3.0 * H100_SQ * a * a * V_phi

    @derivative("dphi")
    def d_dphi(self, ddphi):
        return ddphi

    @derivative("ddphi")
    def d_ddphi(self, ddphi, dphi, aH, k, a, V_phiphi, h_prime, phidot):
        return (
            -2.0 * aH * ddphi
            - (k * k + 3.0 * H100_SQ * a * a * V_phiphi) * dphi
            - 0.5 * h_prime * phidot
        )


QCDM = LCDM.swap(CosmologicalConstant, Quintessence, name="QCDM")


QCDM_COSMOLOGY = {
    "Omega_m": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omega_b": 0.02238280 / 0.6732117**2,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "k_p": 0.05,
    "H_0": 67.32117,
    "T_cmb": 2.7255,
    "Y_He": 0.2454006,
    "N_eff": 3.046,
}
"""The LCDM cosmology the field's are made from (:func:`qcdm_parameters`)."""


LAM = 1.0


def qcdm_parameters(lam=0.0, V0=None):
    """QCDM's own parameters: LCDM's, with the potential's height ``V0`` for the dark-energy density.

    By default ``V0`` is that LCDM's ``omega_de``, so a flat potential (``lam =
    0``) is the same universe. No budget closes: a rolling field's Hubble
    constant comes out of the solve.
    """

    lcdm = LCDM.physical_parameters(QCDM_COSMOLOGY)
    omega_de = lcdm.pop("omega_de")
    return {**lcdm, "lam": lam, "V0": omega_de if V0 is None else V0}


# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


REFERENCE_COSMOLOGY = {
    "Omega_m": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omega_b": 0.02238280 / 0.6732117**2,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omega_k": 0.0,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H_0": 67.32117,
    "T_cmb": 2.7255,
    "Y_He": 0.2454006,
    "N_eff": 3.046,
    "N_mnu": 0.0,
    "m_nu": 0.0,
    "k_p": 0.05,
}


MATTER_POWER_K = np.geomspace(1.0e-4, 1.0, 128, dtype=np.float64)


BATCH_SIZE_CASES = [pytest.param(1, id="N1"), pytest.param(128, id="N128")]


def _perturbed_cosmologies(n):
    """Return deterministic nearby flat LCDM parameter dictionaries."""

    if n == 1:
        return [REFERENCE_COSMOLOGY.copy()]
    phase = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    cosmologies = []
    for ph in phase:
        param = REFERENCE_COSMOLOGY.copy()
        omega_b_h2 = 0.02238280 * (1.0 + 0.006 * ph)
        omega_c_h2 = 0.1201075 * (1.0 - 0.008 * ph)
        h = 0.6732117 * (1.0 + 0.004 * np.sin(np.pi * ph))
        param.update(
            {
                "Omega_m": (omega_b_h2 + omega_c_h2) / h**2,
                "Omega_b": omega_b_h2 / h**2,
                "H_0": 100.0 * h,
                "Y_He": REFERENCE_COSMOLOGY["Y_He"]
                * (1.0 + 0.003 * np.cos(0.5 * np.pi * ph)),
                "T_cmb": REFERENCE_COSMOLOGY["T_cmb"] * (1.0 + 0.0015 * ph),
                "N_eff": REFERENCE_COSMOLOGY["N_eff"]
                * (1.0 + 0.002 * np.sin(2.0 * np.pi * ph)),
            }
        )
        cosmologies.append(param)
    return cosmologies


def _class_base(param):
    """The CLASS settings of a flat, massless-neutrino ``param``, shared by every CLASS reference."""

    h = param["H_0"] / 100.0
    return {
        "h": h,
        "omega_b": param["Omega_b"] * h**2,
        "omega_cdm": (param["Omega_m"] - param["Omega_b"]) * h**2,
        "T_cmb": param["T_cmb"],
        "YHe": param["Y_He"],
        "N_ur": param["N_eff"],
        "N_ncdm": 0,
        "Omega_k": 0.0,
    }


def by_name(values, keep):
    """Split ``solve_eb``'s values into ``{name: (n_cosmologies, n_k, n_tau)}``."""
    return dict(zip(keep, np.moveaxis(np.asarray(values), 1, 0)))


def _matter_power(k_values, batch, a=1.0, **options):
    """``P(k)`` of a cosmology or a list of them as a user computes it: ``solve_eb``, then the spectrum.

    At the scale factor ``a``, today by default.
    """

    import jax
    import jax.numpy as jnp

    from discoeb.models import Model
    from discoeb.observables.power_spectrum import (
        matter_power_spectrum,
    )
    from discoeb.solver._solver import solve_eb

    cosmologies = [batch] if isinstance(batch, Model) else list(batch)
    options.setdefault("thermo_weight", 0.1)
    keep = ("delta_c", "delta_b")
    values = solve_eb(batch, k_values, [a], keep=keep, **options)
    today = dict(zip(keep, jnp.unstack(values[..., -1], axis=1)))
    params = {
        key: jnp.stack([jnp.asarray(getattr(c, key)) for c in cosmologies])
        for key in ("omega_c", "omega_b", "A_s", "n_s", "k_p")
    }
    batched = jax.vmap(matter_power_spectrum, in_axes=(None,))
    return batched(k_values, **today, **params)


# The parameters of the densities, which the derivative tests below have
# always covered. The others have a test each (test_more_parameters_*).
DENSITY_PARAMETERS = ("Omega_m", "Omega_b", "H_0", "T_cmb", "N_eff")


SOLVE_EB_K = MATTER_POWER_K[::16][::-1]  # descending, to exercise the reordering


# The parameters beyond the densities': (model, fiducial changes, P(k) step).
# Both comparisons difference converged solves (rtol = atol = 1e-6): the
# solve is not smooth in the parameters below ~1e-6 of a value, so the
# differences' noise falls as 1 / step (measured for Y_He, 2026-11-13).
MORE_PARAMETERS = {
    "w_DE_0": ("W0WACDM", {"w_DE_0": -0.9, "w_DE_a": 0.1}, 9.0e-3),
    "w_DE_a": ("W0WACDM", {"w_DE_0": -0.9, "w_DE_a": 0.1}, 4.0e-3),
    "Omega_k": ("LCDM", {"Omega_k": 0.02}, 2.0e-3),
    "m_nu": ("LCDM_MNU", {"N_mnu": 1.0, "m_nu": 0.1, "N_eff": 2.0328}, 4.0e-3),
    # z_re moves sum P(k) by 1.4e-5 of itself per unit: a step of 0.05 sits
    # in the converged solve's noise (2026-11-16)
    "z_re": ("LCDM_REIO", {"z_re": 7.67}, 1.0),
}


def _more_parameter_case(name):
    import discoeb.models as models

    model_name, changes, step = MORE_PARAMETERS[name]
    cosmology = {**REFERENCE_COSMOLOGY, **changes}
    model = globals().get(model_name) or getattr(models, model_name)
    return model, cosmology, step


def ty_executable() -> str | None:
    """The ty beside this interpreter (the dev group), else the one on the path."""
    import shutil
    import sys
    from pathlib import Path

    beside = Path(sys.executable).parent / "ty"
    return str(beside) if beside.exists() else shutil.which("ty")


def ty_errors(code: str, tmp_path) -> tuple[set[tuple[int, str]], str]:
    """``{(line, rule)}`` of the errors ty reports in ``code``, and its output."""
    import subprocess
    import sys

    user = tmp_path / "user.py"
    user.write_text(code)
    result = subprocess.run(
        [
            str(ty_executable()),
            "check",
            "--python",
            sys.executable,
            # the package under test, wherever it was imported from
            "--extra-search-path",
            str(Path(discoeb.__file__).parents[1]),
            "--output-format",
            "concise",
            str(user),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    errors = [line for line in result.stdout.splitlines() if "error[" in line]
    found = {(int(e.split(":")[1]), e.split("error[")[1].split("]")[0]) for e in errors}
    return found, result.stdout
