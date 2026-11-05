import math
import os
from dataclasses import dataclass
from functools import partial
from collections import OrderedDict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .eb_model import (
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
    density_parameters,
    dtauda,
    neutrino_density_ratio,
    perturbation_model,
    perturbation_sparsity,
    setup_background_evolution,
)
from .recombination import (
    AKTHOM_PER_OMEGAB_H0SQ,
    THERMAL_PARAMETERS,
    THERMO_ERROR_WEIGHT,
    Recombination,
)
from .thermodynamics_recfast import evaluate_thermo



# Hierarchy truncations. These are the settings of the components in
# :mod:`discoeb.eb_model` -- ``DISCO_LMAX_G`` etc. in the environment override
# their defaults when that module is imported -- mirrored here as module-level
# integers so the state-vector layout below and the accelerator-compiled
# right-hand side treat them as compile-time constants. A caller may rebind a
# component's class attribute (``Photons.LMAX_G = 11``) before the first solve,
# since the GPU pipeline builds its layout from the classes at solve time
# (:func:`_layout_for`). At the defaults the flat-LambdaCDM + massless-neutrino
# layout has NVAR == 50: an 11-variable core (the metric, the fluids and the
# lowest multipoles, coupled through the Einstein constraints) bordered by three
# 13-variable tridiagonal free-streaming hierarchies, which is the structure
# :func:`discoeb.eb_model.perturbation_sparsity` traces through the equations
# and the sparse direct solver exploits.
LMAX_G = Photons.LMAX_G
"""Photon temperature hierarchy truncation (``Theta_0 ... Theta_LMAX_G``)."""

LMAX_POL = Photons.LMAX_POL
"""Photon E-mode polarization hierarchy truncation (``E_2 ... E_LMAX_POL``)."""

LMAX_NR = MasslessNeutrinos.LMAX_NR
"""Massless-neutrino hierarchy truncation (``N_0 ... N_LMAX_NR``)."""

LMAX_NU = MassiveNeutrinos.LMAX_NU
"""Massive-neutrino hierarchy truncation (``psi_0 ... psi_LMAX_NU`` per momentum bin)."""

IX_ETAK = 0
"""State-vector index of the metric perturbation ``etak = k eta``."""

IX_CLXC = 1
"""State-vector index of the CDM density contrast ``clxc``."""

IX_CLXB = 2
"""State-vector index of the baryon density contrast ``clxb``."""

IX_VB = 3
"""State-vector index of the baryon velocity ``vb``."""

IX_G = 4
"""Base index of the photon temperature hierarchy (``Theta_l`` at ``IX_G + l``)."""

IX_POL = IX_G + LMAX_G + 1
"""Base index of the photon polarization hierarchy (``E_l`` at ``IX_POL + (l-2)``)."""

IX_R = IX_POL + LMAX_POL - 1
"""Base index of the massless-neutrino hierarchy (``N_l`` at ``IX_R + l``)."""

NVAR = IX_R + LMAX_NR + 1
"""Total length of the perturbation state vector."""


# ============================================================
# MATTER POWER SPECTRUM
# ============================================================


def total_matter_density_contrast(
    grhoc_t: float,
    clxc: float,
    grhob_t: float,
    clxb: float,
) -> float:
    """Return the total-matter density contrast ``delta_m``.

    Density-weighted combination of the CDM and baryon contrasts,

    ``delta_m = (rho_c delta_c + rho_b delta_b) / (rho_c + rho_b)``,

    using the comoving density coefficients (the ``a`` factors cancel).
    """

    return (grhoc_t * clxc + grhob_t * clxb) / (grhoc_t + grhob_t)


def primordial_curvature_power(
    k: float,
    A_s: float,
    n_s: float,
    k_pivot: float,
) -> float:
    """Return the dimensionless primordial curvature power ``P_R(k)``.

    Power-law spectrum ``P_R(k) = A_s (k / k_pivot)^(n_s - 1)``.
    """

    return A_s * (k / k_pivot) ** (n_s - 1.0)


def matter_power_spectrum(
    k: float,
    delta_m: float,
    A_s: float,
    n_s: float,
    k_pivot: float,
) -> float:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3.

    With initial conditions normalized to unit comoving curvature, the evolved
    total-matter contrast ``delta_m(k)`` acts as a transfer function and

    ``P(k) = (2 pi^2 / k^3) P_R(k) delta_m(k)^2``,

    where ``P_R(k)`` is the dimensionless primordial curvature power. The contrast
    ``delta_m`` must be evaluated at the output time (e.g. ``a = 1``).
    """

    return (
        2.0
        * math.pi**2
        / k**3
        * primordial_curvature_power(k, A_s, n_s, k_pivot)
        * delta_m**2
    )
@dataclass(frozen=True)
class PerturbationLayout:
    """Enabled components and state-vector indices of the perturbation hierarchy.

    Attributes
    ----------
    lmaxg, lmaxpol, lmaxr : int
        Photon-temperature, E-mode-polarization, and massless-neutrino hierarchy
        truncations.
    nqmax : int
        Number of massive-neutrino momentum bins. ``0`` disables massive
        neutrinos entirely (no state variables allocated).
    lmaxnu : int
        Massive-neutrino multipole-hierarchy truncation (used only when
        ``nqmax > 0``).
    enable_dark_energy : bool
        Whether the dynamical dark-energy fluid perturbations are evolved.
    """

    lmaxg: int = 15
    lmaxpol: int = 15
    lmaxr: int = 15
    nqmax: int = 0
    lmaxnu: int = 15
    enable_dark_energy: bool = False

    def __post_init__(self):
        if self.lmaxg < 2 or self.lmaxpol < 2 or self.lmaxr < 2:
            raise ValueError("lmaxg, lmaxpol, lmaxr must each be >= 2")
        if self.nqmax < 0:
            raise ValueError("nqmax must be non-negative")
        if self.nqmax > 0 and self.lmaxnu < 2:
            raise ValueError("lmaxnu must be >= 2 when massive neutrinos are enabled")

    # --- Core (always present) -------------------------------------------------

    @property
    def ix_etak(self) -> int:
        """Index of the synchronous-gauge metric perturbation ``etak = k eta``."""
        return 0

    @property
    def ix_clxc(self) -> int:
        """Index of the CDM density contrast."""
        return 1

    @property
    def ix_clxb(self) -> int:
        """Index of the baryon density contrast."""
        return 2

    @property
    def ix_vb(self) -> int:
        """Index of the baryon velocity."""
        return 3

    @property
    def ix_g(self) -> int:
        """Base index of the photon temperature hierarchy (``Theta_l`` at ix_g + l)."""
        return 4

    @property
    def ix_pol(self) -> int:
        """Base index of the polarization hierarchy (``E_l`` at ix_pol + (l - 2))."""
        return self.ix_g + self.lmaxg + 1

    @property
    def ix_r(self) -> int:
        """Base index of the massless-neutrino hierarchy (``N_l`` at ix_r + l)."""
        return self.ix_pol + self.lmaxpol - 1

    @property
    def _ix_after_massless(self) -> int:
        """First index after the always-present core hierarchies."""
        return self.ix_r + self.lmaxr + 1

    # --- Optional: dynamical dark energy --------------------------------------

    @property
    def has_dark_energy(self) -> bool:
        return self.enable_dark_energy

    @property
    def ix_clxq(self) -> int:
        """Index of the dark-energy density contrast (only if dark energy enabled)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless

    @property
    def ix_thetaq(self) -> int:
        """Index of the dark-energy velocity (only if dark energy enabled)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless + 1

    @property
    def _ix_after_dark_energy(self) -> int:
        return self._ix_after_massless + (2 if self.enable_dark_energy else 0)

    # --- Optional: massive neutrinos ------------------------------------------

    @property
    def has_massive_neutrinos(self) -> bool:
        return self.nqmax > 0

    @property
    def ix_massive_nu(self) -> int:
        """Base index of the massive-neutrino momentum-bin hierarchies.

        The multipole ``psi_l`` of momentum bin ``q`` is stored at
        ``ix_massive_nu + q * (lmaxnu + 1) + l`` (**bin-major, multipole-minor**),
        so that each bin's free-streaming tail ``psi_3 ... psi_lmaxnu`` is
        contiguous and is one tridiagonal block of the sparsity pattern
        (:func:`discoeb.eb_model.perturbation_sparsity`).
        """
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        return self._ix_after_dark_energy

    def ix_psi_base(self, q: int) -> int:
        """Return the state index of ``psi_0`` for momentum bin ``q``."""
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        if not (0 <= q < self.nqmax):
            raise IndexError(f"momentum bin q={q} out of range [0, {self.nqmax})")
        return self.ix_massive_nu + q * (self.lmaxnu + 1)

    def ix_psi(self, l: int, q: int) -> int:
        """Return the state index of massive-neutrino multipole ``psi_l`` of bin ``q``."""
        if not (0 <= l <= self.lmaxnu):
            raise IndexError(f"multipole l={l} out of range [0, {self.lmaxnu}]")
        return self.ix_psi_base(q) + l

    # --- Total size -----------------------------------------------------------

    @property
    def nvar(self) -> int:
        """Total length of the trimmed perturbation state vector."""
        n = self._ix_after_dark_energy
        if self.nqmax > 0:
            n += self.nqmax * (self.lmaxnu + 1)
        return n

    @classmethod
    def from_cosmology(
        cls,
        cosmology,
        *,
        lmaxg: int = 15,
        lmaxpol: int = 15,
        lmaxr: int = 15,
        lmaxnu: int = 15,
        nqmax: int = 5,
    ) -> "PerturbationLayout":
        """Return the trimmed layout implied by a cosmology's enabled components.

        Dynamical dark energy is enabled when ``w_DE_0 != -1`` or ``w_DE_a != 0``;
        massive neutrinos are enabled (with ``nqmax`` momentum bins) when
        ``num_massive_neutrinos > 0``. Curvature does not affect the layout.
        """

        enable_de = (cosmology.w_DE_0 != -1.0) or (cosmology.w_DE_a != 0.0)
        nqmax_massive = nqmax if cosmology.num_massive_neutrinos > 0 else 0
        return cls(
            lmaxg=lmaxg,
            lmaxpol=lmaxpol,
            lmaxr=lmaxr,
            nqmax=nqmax_massive,
            lmaxnu=lmaxnu,
            enable_dark_energy=enable_de,
        )


FLAT_MASSLESS_LAYOUT = PerturbationLayout()
"""Default flat-LambdaCDM + massless-neutrino layout (NVAR == 50).

Reproduces the fixed index constants of :mod:`discoeb.perturbations`.
"""
class _ParamCosmology(NamedTuple):
    """Hashable view of the historical DISCO-EB parameter dictionary."""

    Omegam: float
    Omegab: float
    w_DE_0: float
    w_DE_a: float
    cs2_DE: float
    Omegak: float
    A_s: float
    n_s: float
    H0: float
    T_cmb: float
    Y_He: float
    Neff_massless: float
    num_massive_neutrinos: float
    mnu: float
    k_pivot: float
    tau_reio: float = 0.0  # reionization optical depth (0: no reionization)

    @property
    def h(self):
        return self.H0 / 100.0

    @property
    def Omegac(self):
        return self.Omegam - self.Omegab

    @property
    def omega_b_h2(self):
        return self.Omegab * self.h**2

    @property
    def omega_c_h2(self):
        return self.Omegac * self.h**2

    @property
    def tau_reion(self):
        return self.tau_reio

    @property
    def N_eff(self):
        return self.Neff_massless + self.num_massive_neutrinos

    def to_background_params(self):
        return {
            "Omegam": self.Omegam,
            "Omegab": self.Omegab,
            "w_DE_0": self.w_DE_0,
            "w_DE_a": self.w_DE_a,
            "cs2_DE": self.cs2_DE,
            "Omegak": self.Omegak,
            "A_s": self.A_s,
            "n_s": self.n_s,
            "H0": self.H0,
            "Tcmb": self.T_cmb,
            "YHe": self.Y_He,
            "Neff": self.Neff_massless,
            "Nmnu": self.num_massive_neutrinos,
            "mnu": self.mnu,
            "tau_reio": self.tau_reio,
        }


def _as_cosmology(cosmology):
    """Normalize a legacy parameter dictionary for the accelerator path."""

    if not isinstance(cosmology, dict):
        return cosmology
    return _ParamCosmology(
        Omegam=float(cosmology["Omegam"]),
        Omegab=float(cosmology["Omegab"]),
        w_DE_0=float(cosmology.get("w_DE_0", -1.0)),
        w_DE_a=float(cosmology.get("w_DE_a", 0.0)),
        cs2_DE=float(cosmology.get("cs2_DE", 1.0)),
        Omegak=float(cosmology.get("Omegak", 0.0)),
        A_s=float(cosmology["A_s"]),
        n_s=float(cosmology["n_s"]),
        H0=float(cosmology["H0"]),
        T_cmb=float(cosmology["Tcmb"]),
        Y_He=float(cosmology["YHe"]),
        Neff_massless=float(cosmology["Neff"]),
        num_massive_neutrinos=float(cosmology.get("Nmnu", 0.0)),
        mnu=float(cosmology.get("mnu", 0.0)),
        k_pivot=float(cosmology.get("k_p", 0.05)),
        tau_reio=float(cosmology.get("tau_reio", 0.0)),
    )

# Solver and grid presets.
N_THERMO_GRID = 2048
"""Points of the adaptive RECFAST grid of :func:`thermal_history_batch` and :func:`host_thermal_history`."""
N_BACKGROUND_GRID = 4096
THERMO_BATCH_CHUNK = 128
"""Cosmologies per lane-padded RECFAST launch in :func:`thermal_history_batch`."""
PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = 1.0e-2
PERTURB_MAX_STEPS = 20000
TAU_START = 0.1

# Packed-parameter row layout consumed by the device rhs/jac.
#   p = (grhog, grhornomass, grhoc, grhob, grhov,
#        tau_start, tau_end, k, cosmology_idx, save_index,
#        w_DE_0, w_DE_a, cs2_DE, grhok, grhor_nu, n_mnu, amnu,
#        eps, dir_grhog, dir_grhornomass, dir_grhoc, dir_grhob, dir_grhov,
#        YHe, Tcmb, H0, Omegab, z_re)
# grhok = grhom * Omega_k is the (constant) comoving curvature density
# coefficient 8*pi*G*rho_K a^2; it is 0 for a flat cosmology. The rhs reads
# density ``c`` as ``p[c] + p[IX_EPS] * p[IX_DIR + c]``; ``eps`` is zero in
# every solve and the ``dir`` columns are kept for callers that seed a
# direction of their own. The last five columns are what the recombination
# rates and reionization read (:data:`discoeb.recombination.THERMAL_PARAMETERS`).
IX_TAU_START = 5
IX_TAU_END = 6
IX_K = 7
IX_COSMOLOGY = 8
IX_SAVE_INDEX = 9
IX_W_DE_0 = 10
IX_W_DE_A = 11
IX_CS2_DE = 12
IX_GRHOK = 13
IX_GRHOR_NU = 14
IX_NMNU = 15
IX_AMNU = 16
IX_EPS = 17
IX_DIR = 18
N_PARAM_HIERARCHY = IX_DIR + 5
IX_YHE = N_PARAM_HIERARCHY
IX_TCMB = N_PARAM_HIERARCHY + 1
IX_H0 = N_PARAM_HIERARCHY + 2
IX_OMEGAB = N_PARAM_HIERARCHY + 3
IX_ZRE = N_PARAM_HIERARCHY + 4
N_PARAM = N_PARAM_HIERARCHY + len(THERMAL_PARAMETERS)

PARAM_COLUMNS = {
    "grhog_nominal": 0,
    "grhornomass_nominal": 1,
    "grhoc_nominal": 2,
    "grhob_nominal": 3,
    "grhov_nominal": 4,
    "tau_start": IX_TAU_START,
    "tau_end": IX_TAU_END,
    "k": IX_K,
    "cosmology_tag": IX_COSMOLOGY,
    "save_index": IX_SAVE_INDEX,
    "w_DE_0": IX_W_DE_0,
    "w_DE_a": IX_W_DE_A,
    "cs2_DE": IX_CS2_DE,
    "grhok": IX_GRHOK,
    "grhor_nu": IX_GRHOR_NU,
    "n_mnu": IX_NMNU,
    "amnu": IX_AMNU,
    "eps": IX_EPS,
    "dir_grhog": IX_DIR,
    "dir_grhornomass": IX_DIR + 1,
    "dir_grhoc": IX_DIR + 2,
    "dir_grhob": IX_DIR + 3,
    "dir_grhov": IX_DIR + 4,
    **{name: N_PARAM_HIERARCHY + i for i, name in enumerate(THERMAL_PARAMETERS)},
}
"""The row's columns by the names the component model's providers take them under
(:mod:`discoeb.eb_model`, :mod:`discoeb.recombination`): ``grhog_nominal`` is
read into the equations as ``grhog = grhog_nominal + eps * dir_grhog``."""

PHYSICAL_DENSITY_COLUMNS = (0, 1, 2, 3, 4)
"""Packed-parameter columns carrying the background densities
``(grhog, grhornomass, grhoc, grhob, grhov)``."""

THERMAL_COLUMNS = (IX_YHE, IX_TCMB, IX_H0, IX_OMEGAB)
"""Packed-parameter columns the recombination rates read: ``(YHe, Tcmb, H0, Omegab)``."""

PHYSICAL_COLUMNS = PHYSICAL_DENSITY_COLUMNS + THERMAL_COLUMNS
"""Every column a derivative with respect to the cosmology wants."""

TRAJECTORIES_PER_BLOCK = 32
"""Wave modes packed into one CUDA block by modax's thread-local Rodas5P kernel."""

CUDA_MAX_REGISTERS = int(os.environ.get("DISCO_CUDA_MAX_REGISTERS", "168"))
"""Per-thread register cap for that kernel: fewer registers, more resident warps."""

NQMAX = MassiveNeutrinos.NQMAX
"""Massive-neutrino momentum bins used by both the background and the hierarchy."""

THERMO_RTOL = 1e-5
THERMO_ATOL = 1e-7
"""Tolerances of the stand-alone RECFAST solve of :func:`thermal_history_batch`."""


class BoundedCache(OrderedDict):
    """A least-recently-used ``dict`` holding at most ``maxsize`` entries."""

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = int(maxsize)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            self.popitem(last=False)


BATCH_CACHE_SIZE = 8
"""Batches whose prepared solve and packed initial states are kept."""

_DENSITY_CACHE: BoundedCache = BoundedCache(4096)
_PREPARED_SOLVE_CACHE: BoundedCache = BoundedCache(BATCH_CACHE_SIZE)
_PACKED_BATCH_CACHE: BoundedCache = BoundedCache(BATCH_CACHE_SIZE)
_KERNELS: dict = {}


def _densities(cosmology) -> dict:
    """:func:`discoeb.eb_model.density_parameters` of a cosmology, as floats, memoised."""

    cosmology = _as_cosmology(cosmology)
    cached = _DENSITY_CACHE.get(cosmology)
    if cached is None:
        d = density_parameters(cosmology.to_background_params(), xp=np)
        cached = _DENSITY_CACHE[cosmology] = {k: float(v) for k, v in d.items()}
    return cached


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """``(grhog, grhornomass, grhoc, grhob, grhov)`` of :func:`discoeb.eb_model.density_parameters`."""

    d = _densities(cosmology)
    return (d["grhog"], d["grhornomass"], d["grhoc"], d["grhob"], d["grhov"])


# ============================================================
# THE JOINT MODEL AND ITS KERNEL
# ============================================================


def _layout_for(cosmology) -> PerturbationLayout:
    """The perturbation layout the module presets imply for a cosmology."""

    return PerturbationLayout.from_cosmology(
        cosmology,
        lmaxg=Photons.LMAX_G,
        lmaxpol=Photons.LMAX_POL,
        lmaxr=MasslessNeutrinos.LMAX_NR,
        lmaxnu=MassiveNeutrinos.LMAX_NU,
        nqmax=MassiveNeutrinos.NQMAX,
    )


def joint_model(layout, extra=()):
    """The hierarchy of ``layout`` with :class:`~discoeb.recombination.Recombination` appended.

    The recombination states come *after* the hierarchy's, so the layout's
    indices (:data:`IX_ETAK` ...) hold and the thermal states are the last
    five columns of every history: ``ln a``, ``x_H``, ``x_He``, ``T_m`` and
    the optical depth, at ``model.index["lna"]`` and so on.
    """

    return perturbation_model(layout, extra=[Recombination(), *extra])


def thermal_indices(model) -> tuple[int, ...]:
    """The flat indices of the five recombination states in ``model``."""

    return tuple(model.index[n] for n in ("lna", "thermo_0", "thermo_1", "thermo_2", "kappa"))


def joint_kernel(layout, extra=(), key=None):
    """``(model, rhs, sparsity)`` of the joint model, built once per layout (and ``key``).

    The right-hand side closes over nothing that varies between batches, so
    modax compiles it once; the pattern is the hierarchy's with the five
    driver columns in every row and the driver rows on their own, which
    modax's ``split=True`` reads as a driver block and a main block.
    """

    cache_key = (layout, key)
    cached = _KERNELS.get(cache_key)
    if cached is None:
        model = joint_model(layout, extra)
        rhs = model.build(PARAM_COLUMNS)
        cached = _KERNELS[cache_key] = (model, rhs, model.sparsity())
    return cached


def _stack_background_params(cosmologies) -> dict:
    """The cosmologies' background parameters stacked by name, one vmap lane each."""

    rows = [c.to_background_params() for c in cosmologies]
    return {key: jnp.asarray([row[key] for row in rows], dtype=jnp.float64) for key in rows[0]}


def _background_tau(d):
    """``(a_grid, tau_grid)`` of the conformal-time quadrature for densities ``d`` (JAX)."""

    a_bg = jnp.geomspace(1.0e-9, 1.0, N_BACKGROUND_GRID)
    dt = jax.vmap(dtauda, in_axes=(0, None))(a_bg, d)
    steps = 0.5 * (dt[1:] + dt[:-1]) * jnp.diff(a_bg)
    radiation = (
        d["grhog"] + d["grhornomass"]
        + d["grhor_nu"] * d["n_mnu"] * neutrino_density_ratio(1.0e-9, d["amnu"])
    )
    taumin = 1.0e-9 / jnp.sqrt(radiation / 3.0)
    return a_bg, jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(steps))) + taumin


def _reionization_redshift(a_bg, tau_bg, xe_rec, params, tau_reio):
    """``z_re`` of CAMB's step, by bisection on the background grid, for ``tau_reio > 0``."""

    f_he = params["YHe"] / (4.0 * (1.0 - params["YHe"]))
    akthom = AKTHOM_PER_OMEGAB_H0SQ * (1.0 - params["YHe"]) * params["Omegab"] * params["H0"] ** 2
    dtau = jnp.gradient(tau_bg)

    def xe_reio(a, z_re):
        z = 1.0 / a - 1.0
        y, y_re = (1.0 + z) ** 1.5, (1.0 + z_re) ** 1.5
        delta_y = 1.5 * jnp.sqrt(1.0 + z_re) * 0.5
        step = 0.5 * (1.0 + jnp.tanh((y_re - y) / delta_y))
        he3 = 0.5 * (1.0 + jnp.tanh((3.5 - z) / 0.5))
        return (1.0 + f_he - xe_rec) * step + f_he * he3

    def tau_of(z_re):
        return jnp.sum(xe_reio(a_bg, z_re) * akthom / a_bg**2 * dtau)

    def bisect(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        too_much = tau_of(mid) > tau_reio
        return (jnp.where(too_much, lo, mid), jnp.where(too_much, mid, hi)), None

    (lo, hi), _ = jax.lax.scan(bisect, (jnp.asarray(1.0), jnp.asarray(40.0)), None, length=40)
    return jnp.where(tau_reio > 0.0, 0.5 * (lo + hi), 0.0)


@jax.jit
def background_endpoints(stacked):
    """``(tau0, lna_start, z_re)`` of stacked background parameters, one vmap lane each.

    The conformal age, where each trajectory stops; ``ln a`` at
    :data:`TAU_START`, where it starts; and the reionization redshift that
    gives ``tau_reio`` (0 without reionization), solved on the background with
    the residual recombination ionization neglected, which moves ``tau_reio``
    by 2e-4 of itself.
    """

    def one(params):
        d = density_parameters({k: v for k, v in params.items() if k != "tau_reio"})
        a_bg, tau_bg = _background_tau(d)
        lna_start = jnp.log(jnp.interp(TAU_START, tau_bg, a_bg))
        z_re = _reionization_redshift(a_bg, tau_bg, 0.0, params, params["tau_reio"])
        return tau_bg[-1], lna_start, z_re

    return jax.vmap(one)(stacked)


class _PreparedSolve(NamedTuple):
    """What a batch shares: the compiled joint model and each cosmology's endpoints."""

    cosmologies: tuple
    layout: object
    model: object
    ode_fn: object
    sparsity: object
    tau0: np.ndarray  # (n_cosmologies,)
    lna_start: np.ndarray
    z_re: np.ndarray


def _prepare_solve(cosmologies, extra=(), key=None) -> _PreparedSolve:
    """Return (and cache) the joint kernel and the endpoints of a batch of cosmologies.

    All cosmologies in a batch must share a perturbation layout. Nothing here
    depends on the wave modes, the save times or the tolerances; the kernel
    is compiled once per layout and the endpoints are a vmapped quadrature of
    a few milliseconds.
    """

    cosmologies = tuple(_as_cosmology(cosmology) for cosmology in cosmologies)
    cache_key = (cosmologies, key, MassiveNeutrinos.NQMAX, Photons.LMAX_G, Photons.LMAX_POL,
                 MasslessNeutrinos.LMAX_NR, MassiveNeutrinos.LMAX_NU, TAU_START)
    cached = _PREPARED_SOLVE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    layout = _layout_for(cosmologies[0])
    for cosmology in cosmologies[1:]:
        if _layout_for(cosmology) != layout:
            raise ValueError(
                "all cosmologies in a batch must share a perturbation layout "
                "(same dark-energy / massive-neutrino settings)"
            )
    model, rhs, sparsity = joint_kernel(layout, extra, key)
    tau0, lna_start, z_re = (np.asarray(x) for x in background_endpoints(_stack_background_params(cosmologies)))
    prepared = _PreparedSolve(cosmologies, layout, model, rhs, sparsity, tau0, lna_start, z_re)
    _PREPARED_SOLVE_CACHE[cache_key] = prepared
    return prepared


def make_params(cosmology, k_values: np.ndarray, tau_end: float, z_re: float = 0.0) -> np.ndarray:
    """Return per-mode packed parameters for the device rhs, one row per wave mode."""

    cosmology = _as_cosmology(cosmology)
    d = _densities(cosmology)
    k_values = np.asarray(k_values, dtype=np.float64)
    row = {
        "grhog_nominal": d["grhog"],
        "grhornomass_nominal": d["grhornomass"],
        "grhoc_nominal": d["grhoc"],
        "grhob_nominal": d["grhob"],
        "grhov_nominal": d["grhov"],
        "tau_start": TAU_START,
        "tau_end": tau_end,
        "save_index": 1.0,
        "w_DE_0": cosmology.w_DE_0,
        "w_DE_a": cosmology.w_DE_a,
        "cs2_DE": cosmology.cs2_DE,
        "grhok": d["grhok"],
        "grhor_nu": d["grhor_nu"],
        "n_mnu": d["n_mnu"],
        "amnu": d["amnu"],
        "YHe": cosmology.Y_He,
        "Tcmb": cosmology.T_cmb,
        "H0": cosmology.H0,
        "Omegab": cosmology.Omegab,
        "z_re": z_re,
    }
    params = np.zeros((len(k_values), N_PARAM), dtype=np.float64)
    for name, column in PARAM_COLUMNS.items():
        params[:, column] = row.get(name, 0.0)
    params[:, IX_K] = k_values
    return params


def make_initial_states(cosmology, k_values: np.ndarray, model, lna_start: float) -> np.ndarray:
    """Adiabatic initial states for every wave mode plus the fully ionized thermal state, ``(n_k, nvar)``."""

    cosmology = _as_cosmology(cosmology)
    grhog, grhornomass, grhoc, grhob, grhov = density_coefficients(cosmology)
    return model.initial_state(
        k_values,
        tau=TAU_START,
        grhog=grhog,
        grhornomass=grhornomass,
        grhoc=grhoc,
        grhob=grhob,
        grhov=grhov,
        w_DE_0=cosmology.w_DE_0,
        w_DE_a=cosmology.w_DE_a,
        lna_start=float(lna_start),
        Tcmb=cosmology.T_cmb,
    )


def _pack_batch(prepared, k_values, tau_save=None):
    """Return (and cache) the packed ``(params, y0, t_span, order, k_sorted)`` of a batch.

    Trajectories are packed as ``row = k_idx * n_cosmo + cosmology_idx``, so a
    block of consecutive rows shares one wave mode; ``tau_save`` is the shared
    grid of save times (default: the two endpoints), with every trajectory
    stopping at its own ``tau0`` through ``IX_TAU_END``.
    """

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    tau_key = None if tau_save is None else np.asarray(tau_save, np.float64).tobytes()
    key = (prepared.cosmologies, k_arr.shape, k_arr.tobytes(), tau_key)
    cached = _PACKED_BATCH_CACHE.get(key)
    if cached is not None:
        return cached

    model = prepared.model
    n_cosmo = len(prepared.cosmologies)
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]
    n_traj = len(k_sorted) * n_cosmo
    params = np.empty((n_traj, N_PARAM), dtype=np.float64)
    y0 = np.empty((n_traj, model.nvar), dtype=np.float64)
    for ci, cosmology in enumerate(prepared.cosmologies):
        base = make_params(cosmology, k_sorted, float(prepared.tau0[ci]), float(prepared.z_re[ci]))
        base[:, IX_COSMOLOGY] = float(ci)
        params[ci::n_cosmo] = base
        y0[ci::n_cosmo] = make_initial_states(cosmology, k_sorted, model, prepared.lna_start[ci])
    if tau_save is None:
        t_span = np.asarray((TAU_START, float(np.max(prepared.tau0))), dtype=np.float64)
    else:
        t_span = np.ascontiguousarray(np.asarray(tau_save, dtype=np.float64))
    packed = (np.ascontiguousarray(params), np.ascontiguousarray(y0), t_span, order, k_sorted)
    _PACKED_BATCH_CACHE[key] = packed
    return packed


def error_weights_for(prepared, n_traj: int, base=None, thermo_weight: float = THERMO_ERROR_WEIGHT):
    """The solver's error weights ``(n_traj, nvar)``: ``base`` (or ones) for the hierarchy,
    ``thermo_weight`` on ``ln a`` and the RECFAST states, zero on the optical depth."""

    model = prepared.model
    weights = np.ones((n_traj, model.nvar), dtype=np.float64)
    if base is not None:
        base = np.asarray(base, dtype=np.float64)
        weights[:, : base.shape[-1]] = base
    lna, x_h, x_he, t_m, kappa = thermal_indices(model)
    weights[:, [lna, x_h, x_he, t_m]] = thermo_weight
    weights[:, kappa] = 0.0
    return weights


def _launch(prepared, params, y0, t_span, *, rtol, atol, first_step, max_steps,
            error_weights=None, thermo_weight=THERMO_ERROR_WEIGHT, lu_precision="fp32", **extra):
    """One split modax Rodas5P launch of the prepared solve over the packed trajectories."""

    from modax.rodas5P import solve as rodas5P_solve

    return rodas5P_solve(
        prepared.ode_fn,
        y0,
        t_span,
        params,
        sparsity=prepared.sparsity,
        lu_precision=lu_precision,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
        error_weights=error_weights_for(prepared, np.shape(params)[0], error_weights, thermo_weight),
        split=True,
        **extra,
    )


def solve_perturbation_history(
    k_values,
    cosmology,
    *,
    tau_save=None,
    as_jax: bool = False,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve the joint system of one cosmology and return the saved history.

    Returns ``(history, layout, model)``: ``history`` has shape
    ``(n_k, len(tau_save), nvar)`` in the caller's ``k`` order, with the
    hierarchy at the layout's indices and the five recombination states last
    (:func:`thermal_indices`).
    """

    hist, k_sorted, prepared = solve_perturbation_history_batch(
        k_values, (cosmology,), tau_save=tau_save, rtol=rtol, atol=atol,
        first_step=first_step, max_steps=max_steps,
    )
    inv_order = np.argsort(np.argsort(np.asarray(k_values, dtype=np.float64), kind="stable"))
    history = hist[:, 0][jnp.asarray(inv_order)]
    if not as_jax:
        history = np.asarray(history)
    return history, prepared.layout, prepared.model


def solve_perturbation_history_batch(
    k_values,
    cosmologies,
    *,
    tau_save=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve a batch of cosmologies on a shared ``tau_save`` grid in one launch.

    All ``n_cosmologies * n_k`` trajectories are integrated by a single modax
    launch, the recombination of each one ahead of its hierarchy in every step
    (modax's ``split``). Each trajectory stops at its *own* ``tau0``; save
    times beyond it hold its frozen final state. Returns
    ``(history, k_sorted, prepared)`` with ``history`` a device-resident
    ``(n_k, n_cosmologies, len(tau_save), nvar)`` in ascending ``k``.
    """

    prepared = _prepare_solve(tuple(cosmologies))
    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_save)
    sol = _launch(prepared, params, y0, t_span, rtol=rtol, atol=atol, first_step=first_step, max_steps=max_steps)
    hist = jnp.asarray(sol).reshape(len(k_sorted), len(prepared.cosmologies), len(t_span), prepared.model.nvar)
    return hist, k_sorted, prepared


def solve_perturbations(k_values, cosmology, **solve_kwargs):
    """Final states ``(n_k, nvar)`` of one cosmology at ``tau0``."""

    solve_kwargs.pop("tau_save", None)
    hist, _, _ = solve_perturbation_history(k_values, cosmology, **solve_kwargs)
    return hist[:, -1, :]


def solve_matter_power_spectrum(k_values, cosmology, **solve_kwargs) -> np.ndarray:
    """Return the linear matter power spectrum ``P(k)`` in Mpc^3 for one cosmology."""

    return solve_matter_power_spectrum_batch(k_values, (cosmology,), **solve_kwargs)[0]


def _power_spectrum_from_final(y_final, k_sorted, cosmologies, k_values):
    """``P(k)`` ``(n_cosmo, n_k)`` in the caller's order from final states ``(n_k, n_cosmo, nvar)``."""

    dens = np.array([density_coefficients(c) for c in cosmologies])
    grhoc, grhob = dens[:, 2][None, :], dens[:, 3][None, :]
    delta_m = (grhoc * y_final[:, :, IX_CLXC] + grhob * y_final[:, :, IX_CLXB]) / (grhoc + grhob)
    A_s = np.array([c.A_s for c in cosmologies])[None, :]
    n_s = np.array([c.n_s for c in cosmologies])[None, :]
    k_p = np.array([c.k_pivot for c in cosmologies])[None, :]
    ks = k_sorted[:, None]
    pk_sorted = 2.0 * np.pi**2 / ks**3 * A_s * (ks / k_p) ** (n_s - 1.0) * delta_m**2
    order = np.argsort(np.asarray(k_values, dtype=np.float64), kind="stable")
    pk = np.empty_like(pk_sorted.T)
    pk[:, order] = pk_sorted.T
    return pk


def solve_matter_power_spectrum_batch(
    k_values,
    cosmologies,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
) -> np.ndarray:
    """Return ``P(k)`` in Mpc^3, shape ``(n_cosmologies, n_k)``, for a batch in one launch."""

    prepared = _prepare_solve(tuple(cosmologies))
    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values)
    sol = _launch(prepared, params, y0, t_span, rtol=rtol, atol=atol, first_step=first_step, max_steps=max_steps)
    y_final = np.asarray(sol)[:, -1, :].reshape(len(k_sorted), len(prepared.cosmologies), prepared.model.nvar)
    return _power_spectrum_from_final(y_final, k_sorted, prepared.cosmologies, k_values)


# ============================================================
# DIFFERENTIABLE SOLVES
# ============================================================


def _row_by_name(params_row, densities=None, thermal=None) -> dict:
    """The parameter row by name, with the traced densities and thermal inputs in place."""

    row = {name: float(params_row[column]) for name, column in PARAM_COLUMNS.items()}
    if densities is not None:
        for name, column in PARAM_COLUMNS.items():
            if column in PHYSICAL_DENSITY_COLUMNS:
                row[name] = densities[column]
    if thermal is not None:
        for i, name in enumerate(THERMAL_PARAMETERS[:4]):
            row[name] = thermal[i]
    return row


def solve_history_jax(
    prepared,
    k_values,
    densities=None,
    thermal=None,
    tau_save=None,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    sens_error_control: bool = False,
    error_weights=None,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
):
    """The saved history of one cosmology's modes, differentiable in the cosmology.

    ``densities`` is the five-vector of :data:`PHYSICAL_DENSITY_COLUMNS` and
    ``thermal`` the four-vector ``(YHe, Tcmb, H0, Omegab)`` of
    :data:`THERMAL_COLUMNS`, each defaulting to the cosmology's own; the row
    is built from them so that ``jax.grad`` through this function is the
    derivative of the solve with respect to those nine numbers, recombination
    included: modax's forward sensitivities run over both blocks of the split
    solve. The initial state is held fixed (its dependence on the densities is
    of order ``(k tau_start)^2``), and the trajectory stops at the cosmology's
    own ``tau0``; :func:`matter_power_spectrum_jax` adds the end-time term.

    Returns ``(history, densities, thermal, k_sorted, order)`` with ``history``
    ``(n_k, n_save, nvar)`` in ascending ``k``.
    """

    from modax.rodas5P import solve as rodas5P_solve

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_save)
    if densities is None:
        densities = params[0, list(PHYSICAL_DENSITY_COLUMNS)]
    if thermal is None:
        thermal = params[0, list(THERMAL_COLUMNS)]
    densities = jnp.asarray(densities, dtype=jnp.float64)
    thermal = jnp.asarray(thermal, dtype=jnp.float64)
    n_traj = params.shape[0]
    params_j = jnp.asarray(params)
    params_j = params_j.at[:, jnp.asarray(PHYSICAL_DENSITY_COLUMNS)].set(jnp.broadcast_to(densities, (n_traj, 5)))
    params_j = params_j.at[:, jnp.asarray(THERMAL_COLUMNS)].set(jnp.broadcast_to(thermal, (n_traj, 4)))
    history = rodas5P_solve(
        prepared.ode_fn,
        jnp.asarray(y0),
        jnp.asarray(t_span),
        params_j,
        sparsity=prepared.sparsity,
        lu_precision="fp64",
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        pcoeff=0.3,
        icoeff=0.4,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        tf_index=IX_TAU_END,
        max_registers=CUDA_MAX_REGISTERS,
        error_weights=error_weights_for(prepared, n_traj, error_weights, thermo_weight),
        sens_param_columns=PHYSICAL_COLUMNS,
        sens_error_control=sens_error_control,
        split=True,
    )
    return history, densities, thermal, k_sorted, order


def _tau0_of(densities, row):
    """The conformal age as a function of the five densities (JAX), the other background inputs from ``row``."""

    d = {
        "grhog": densities[0], "grhornomass": densities[1], "grhoc": densities[2], "grhob": densities[3],
        "grhov": densities[4], "grhok": row["grhok"], "grhor_nu": row["grhor_nu"], "n_mnu": row["n_mnu"],
        "amnu": row["amnu"], "w_DE_0": row["w_DE_0"], "w_DE_a": row["w_DE_a"], "cs2_DE": row["cs2_DE"],
    }
    return _background_tau(d)[1][-1]


def matter_power_spectrum_jax(
    k_values,
    cosmology,
    densities=None,
    thermal=None,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    sens_error_control: bool = False,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
):
    """``P(k)`` as a JAX array, differentiable in the densities and the thermal inputs.

    ``densities`` (five, :data:`PHYSICAL_DENSITY_COLUMNS`) and ``thermal``
    (``YHe, Tcmb, H0, Omegab``) default to the cosmology's own. The derivative
    is the total one: the densities enter the hierarchy, the recombination
    rates (through the expansion rate), the matter weighting of ``delta_m``
    and the conformal age at which the state is read, and all four paths are
    differentiated -- the last through ``y(tau_0(d)) = y(tau_0^*) +
    y'(tau_0^*) (tau_0(d) - tau_0^*)``, which is zero in value and carries
    ``d tau_0 / d(densities)`` from the background quadrature. What is held
    fixed is the initial state.
    """

    cosmology = _as_cosmology(cosmology)
    prepared = _prepare_solve((cosmology,))
    history, densities, thermal, k_sorted, order = solve_history_jax(
        prepared, k_values, densities, thermal, rtol=rtol, atol=atol, first_step=first_step,
        max_steps=max_steps, sens_error_control=sens_error_control, thermo_weight=thermo_weight,
    )
    y_final = history[:, -1, :]
    params = _pack_batch(prepared, k_values)[0]
    row = _row_by_name(params[0], densities, thermal)

    # The end-time term: the state's rate at tau0 from the model's own
    # equations on the final state, times how tau0 moves with the densities.
    model = prepared.model
    base = dict(row)
    base.update(model.state_values(y_final, xp=jnp))
    base["k"] = jnp.asarray(k_sorted)
    base["tau"] = jnp.full((len(k_sorted),), float(prepared.tau0[0]))
    z, vb = model.evaluate("z", "vb", math=jnp, **base)
    ks = jnp.asarray(k_sorted)
    tau0 = _tau0_of(densities, row)
    dtau = tau0 - jax.lax.stop_gradient(tau0)
    clxc = y_final[:, IX_CLXC] - ks * z * dtau
    clxb = y_final[:, IX_CLXB] - ks * (z + vb) * dtau

    grhoc_val, grhob_val = densities[2], densities[3]
    delta_m = (grhoc_val * clxc + grhob_val * clxb) / (grhoc_val + grhob_val)
    pk_sorted = (
        2.0 * jnp.pi**2 / ks**3 * cosmology.A_s * (ks / cosmology.k_pivot) ** (cosmology.n_s - 1.0) * delta_m**2
    )
    return pk_sorted[jnp.asarray(np.argsort(order))]


# ============================================================
# THE RECOMBINATION HISTORY ON ITS OWN (diagnostics and grid design)
# ============================================================


@partial(jax.jit, static_argnames=("n_grid", "rtol", "atol"))
def _solve_thermal_histories(stacked, n_grid: int, rtol: float = THERMO_RTOL, atol: float = THERMO_ATOL):
    """The RECFAST solve of :func:`discoeb.eb_model.evolve_background` alone, one vmap lane per cosmology."""

    def one(params):
        param = setup_background_evolution(amin=1e-9, amax=1.01, param=dict(params))
        return evaluate_thermo(param=param, num_thermo=n_grid, rtol=rtol, atol=atol)

    return jax.vmap(one)(stacked)


def thermal_history_batch(cosmologies, n_grid: int = N_THERMO_GRID) -> dict:
    """Solve the recombination history of a batch of cosmologies in one stand-alone launch.

    Returns host arrays: ``a`` of shape ``(n_grid + 1,)`` (the adaptive RECFAST
    grid), and ``xe``, ``Tm`` and ``cs2`` of shape ``(n_cosmologies, n_grid + 1)``.
    This is RECFAST on its own, as :mod:`discoeb.thermodynamics_recfast`
    integrates it in ``ln a``; the perturbation solve integrates the same
    equations in conformal time inside its own kernel.
    """

    cosmologies = tuple(_as_cosmology(cosmology) for cosmology in cosmologies)
    padded = cosmologies + (cosmologies[-1],) * (-len(cosmologies) % THERMO_BATCH_CHUNK)
    stacked = _stack_background_params(padded)
    a, cs2, Tm, mu, xe, dxeda = _solve_thermal_histories(stacked, n_grid, rtol=THERMO_RTOL, atol=THERMO_ATOL)
    n = len(cosmologies)
    return {
        "a": np.asarray(a)[0],
        "xe": np.asarray(xe)[:n, :, 0],
        "Tm": np.asarray(Tm)[:n, :, 0],
        "cs2": np.asarray(cs2)[:n, :, 0],
    }


_HOST_HISTORY_CACHE: BoundedCache = BoundedCache(64)


def host_thermal_history(cosmology, n_grid: int = N_THERMO_GRID):
    """``(tau_grid, values, tau0)`` of one cosmology on a uniform ``log tau`` grid, as host arrays.

    ``values`` is ``(3, n_grid)``: ``a``, the opacity ``kappa'`` (with
    reionization) and ``c_s^2`` from the stand-alone RECFAST solve and the
    background quadrature. Used to design the CMB grids and by the
    thermodynamics tests; the perturbation kernel reads none of it.
    """

    cosmology = _as_cosmology(cosmology)
    key = (cosmology, n_grid)
    cached = _HOST_HISTORY_CACHE.get(key)
    if cached is not None:
        return cached
    hist = thermal_history_batch([cosmology], n_grid)
    params = cosmology.to_background_params()
    d = density_parameters({k: v for k, v in params.items() if k != "tau_reio"}, xp=np)
    a_bg, tau_bg = (np.asarray(x) for x in _background_tau({k: jnp.asarray(v) for k, v in d.items()}))
    tau0 = float(tau_bg[-1])
    tau_grid = np.exp(np.linspace(np.log(TAU_START), np.log(tau0), n_grid))
    a_vals = np.interp(tau_grid, tau_bg, a_bg)
    a_rec, xe_rec, cs2_rec = hist["a"], hist["xe"][0], hist["cs2"][0]
    a_recfast_start = 1.0 / 3501.0
    xe_start = float(np.interp(np.log(a_recfast_start), np.log(a_rec), xe_rec))
    xe_vals = np.where(a_vals < a_recfast_start, xe_start, np.interp(np.log(a_vals), np.log(a_rec), xe_rec))
    xe_bg = np.where(a_bg < a_recfast_start, xe_start, np.interp(np.log(a_bg), np.log(a_rec), xe_rec))
    z_re = float(_reionization_redshift(
        jnp.asarray(a_bg), jnp.asarray(tau_bg), jnp.asarray(xe_bg),
        {k: jnp.asarray(v) for k, v in params.items()}, jnp.asarray(params["tau_reio"]),
    ))
    f_he = params["YHe"] / (4.0 * (1.0 - params["YHe"]))
    if z_re > 0.0:
        z = 1.0 / a_vals - 1.0
        y, y_re = (1.0 + z) ** 1.5, (1.0 + z_re) ** 1.5
        step = 0.5 * (1.0 + np.tanh((y_re - y) / (0.75 * np.sqrt(1.0 + z_re))))
        he3 = 0.5 * (1.0 + np.tanh((3.5 - z) / 0.5))
        xe_vals = xe_vals + (1.0 + f_he - xe_vals) * step + f_he * he3
    akthom = AKTHOM_PER_OMEGAB_H0SQ * (1.0 - params["YHe"]) * params["Omegab"] * params["H0"] ** 2
    opacity = xe_vals * akthom / a_vals**2
    cs2 = np.interp(np.log(a_vals), np.log(a_rec), cs2_rec)
    out = (tau_grid, np.stack((a_vals, opacity, cs2)), tau0)
    _HOST_HISTORY_CACHE[key] = out
    return out


def clear_prepared_solve_cache() -> None:
    """Drop the cached prepared and packed solves and the compiled joint models."""

    _PREPARED_SOLVE_CACHE.clear()
    _PACKED_BATCH_CACHE.clear()
    _HOST_HISTORY_CACHE.clear()
    _KERNELS.clear()
