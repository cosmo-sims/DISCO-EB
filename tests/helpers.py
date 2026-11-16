"""What the test files share: host stand-ins, models, components added from outside, reference cosmologies.

Not a test file. ``conftest.py`` sets the environment before this imports
:mod:`discoeb`; the test folders import from here by name (``pythonpath`` in
``pyproject.toml``).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from jax.typing import ArrayLike

from discoeb.components._component import (
    Component,
    Derived,
    Parameter,
    Species,
    State,
)
from discoeb.components.species import (
    CosmologicalConstant,
    Fluid,
    MassiveNeutrinos,
)
from discoeb.models import LCDM, LCDM_MNU, W0WACDM, W0WACDM_MNU
from discoeb.solver._kernel import PARAM_COLUMNS


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
    their defaults.
    """

    if isinstance(params, dict):
        return model.from_dict(params)
    return [model.from_dict(p) for p in params]


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
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
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
        [ix[f"theta_{ell}"] for ell in range(3, model.setting("LMAX_G") + 1)],
        [ix[f"e_{i}"] for i in range(1, model.setting("LMAX_POL") - 1)],
        [ix[f"n_{ell}"] for ell in range(3, model.setting("LMAX_NR") + 1)],
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
    """Pressureless and at rest in synchronous gauge, like CDM: ``clxx' = -k z``."""

    clxx = State(initial=lambda theta: 0.75 * theta[0])

    Omega_x: ArrayLike = Parameter(0.0)
    grhox: ArrayLike = Derived(lambda grhom, Omega_x: grhom * Omega_x, directed=True)

    @classmethod
    def check(cls, Omega_x):
        if Omega_x < 0.0:
            raise ValueError(f"Omega_x = {Omega_x} must not be negative")

    def grhox_t(self, grhox, inv_a):
        return grhox * inv_a

    def grho(self, grhox_t):
        return grhox_t

    def grho_matter(self, grhox_t):
        """It is matter to the growing mode's initial conditions."""
        return grhox_t

    def dgrho(self, grhox_t, clxx):
        return grhox_t * clxx

    def d_clxx(self, z, k):
        return -k * z


LCDM_X = LCDM.add(SecondDarkMatter, name="LCDM+x")


OMEGA_X = 0.08


def _split(cosmology, omega_x=OMEGA_X):
    """The same universe with ``omega_x`` of its cold dark matter in the second fluid."""

    return {**cosmology, "Omegam": cosmology["Omegam"] - omega_x, "Omega_x": omega_x}


# --- A background state added from outside: the equations split the blocks themselves.


class CosmicTime(Component):
    """The cosmic time ``t`` in Mpc: ``dt / dtau = a``."""

    t = State()

    def d_t(self, a):
        return a


LCDM_T = LCDM.add(CosmicTime, name="LCDM+t")


# --- A scalar field from outside the package: background states, and a budget closed by shooting.


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


QCDM_COSMOLOGY = {
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


def _qcdm_densities(model, cosmology):
    from discoeb.models._cosmology import _columns
    from discoeb.solver._background import density_parameters

    columns = _columns([cosmology], held=False, keys=model.parameter_keys())
    return {k: v[0] for k, v in density_parameters(columns, model=model).items()}


# --- GPU perturbation solves against CLASS: the matter power spectrum and its derivatives.


REFERENCE_COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omegak": 0.0,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
    "Nmnu": 0.0,
    "mnu": 0.0,
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
                "Omegam": (omega_b_h2 + omega_c_h2) / h**2,
                "Omegab": omega_b_h2 / h**2,
                "H0": 100.0 * h,
                "YHe": REFERENCE_COSMOLOGY["YHe"]
                * (1.0 + 0.003 * np.cos(0.5 * np.pi * ph)),
                "Tcmb": REFERENCE_COSMOLOGY["Tcmb"] * (1.0 + 0.0015 * ph),
                "Neff": REFERENCE_COSMOLOGY["Neff"]
                * (1.0 + 0.002 * np.sin(2.0 * np.pi * ph)),
            }
        )
        cosmologies.append(param)
    return cosmologies


def _class_base(param):
    """The CLASS settings of a flat, massless-neutrino ``param``, shared by every CLASS reference."""

    h = param["H0"] / 100.0
    return {
        "h": h,
        "omega_b": param["Omegab"] * h**2,
        "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
        "T_cmb": param["Tcmb"],
        "YHe": param["YHe"],
        "N_ur": param["Neff"],
        "N_ncdm": 0,
        "Omega_k": 0.0,
    }


def _matter_power(k_values, batch, tau_today=None, **options):
    """``P(k)`` of a cosmology or a list of them as a user computes it: ``solve_eb``, then the spectrum.

    ``tau_today`` must lie at or past every conformal age. It defaults to the
    batch's oldest, which needs the parameters as numbers.
    """

    import jax.numpy as jnp

    from discoeb.models import Model
    from discoeb.observables.power_spectrum import (
        matter_power_spectrum,
    )
    from discoeb.solver._solver import conformal_ages, solve_eb

    cosmologies = [batch] if isinstance(batch, Model) else list(batch)
    tau0 = conformal_ages(batch)
    if tau_today is None:
        tau_today = tau0.max()
    options.setdefault("thermo_weight", 0.1)
    solution = solve_eb(
        batch, k_values, [tau_today], keep=("clxc", "clxb", "z", "vb"), **options
    )
    today = {name: solution[name][..., -1] for name in solution.names}
    params = {
        key: jnp.stack([jnp.asarray(getattr(c, key)) for c in cosmologies])
        for key in ("Omegam", "Omegab", "A_s", "n_s", "k_p")
    }
    return matter_power_spectrum(k_values, **today, tau0=tau0, **params)


# The parameters of the densities, which the derivative tests below have
# always covered. The others have a test each (test_more_parameters_*).
DENSITY_PARAMETERS = ("Omegam", "Omegab", "H0", "Tcmb", "Neff")


SOLVE_EB_K = MATTER_POWER_K[::16][::-1]  # descending, to exercise the reordering


# The parameters beyond the densities': (model, fiducial changes, P(k) step).
# Both comparisons difference converged solves (rtol = atol = 1e-6): the
# solve is not smooth in the parameters below ~1e-6 of a value, so the
# differences' noise falls as 1 / step (measured for YHe, 2026-11-13).
MORE_PARAMETERS = {
    "w_DE_0": ("W0WACDM", {"w_DE_0": -0.9, "w_DE_a": 0.1}, 9.0e-3),
    "w_DE_a": ("W0WACDM", {"w_DE_0": -0.9, "w_DE_a": 0.1}, 4.0e-3),
    "Omegak": ("LCDM", {"Omegak": 0.02}, 2.0e-3),
    "mnu": ("LCDM_MNU", {"Nmnu": 1.0, "mnu": 0.1, "Neff": 2.0328}, 4.0e-3),
}


def _more_parameter_case(name):
    import discoeb.models as models

    model_name, changes, step = MORE_PARAMETERS[name]
    cosmology = {**REFERENCE_COSMOLOGY, **changes}
    return getattr(models, model_name), cosmology, step


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
