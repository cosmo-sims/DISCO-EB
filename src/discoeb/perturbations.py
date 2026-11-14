import functools
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from .component import Model, _compile
from .eb_model import (
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
    density_parameters,
    dtauda,
    layout_state_names,
    neutrino_density_ratio,
    perturbation_model,
)
from .recombination import (
    THERMAL_PARAMETERS,
    THERMO_ERROR_WEIGHT,
    Recombination,
    recombination_model,
)

IX_CLXC = 1
"""State-vector index of the CDM density contrast ``clxc`` (every layout)."""

IX_CLXB = 2
"""State-vector index of the baryon density contrast ``clxb`` (every layout)."""


@dataclass(frozen=True)
class PerturbationLayout:
    """Enabled components and state-vector indices of the perturbation hierarchy.

    Attributes
    ----------
    lmaxg, lmaxpol, lmaxr : int
        Truncations of the photon temperature, E-mode polarization and
        massless-neutrino hierarchies.
    nqmax : int
        Number of massive-neutrino momentum bins. ``0`` disables massive
        neutrinos and allocates no states for them.
    lmaxnu : int
        Truncation of the massive-neutrino hierarchy (used only when
        ``nqmax > 0``).
    enable_dark_energy : bool
        Evolve the dynamical dark-energy fluid perturbations.
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
    def ix_clxq(self) -> int:
        """Index of the dark-energy density contrast (dark-energy layouts only)."""
        if not self.enable_dark_energy:
            raise AttributeError("dark energy is not enabled in this layout")
        return self._ix_after_massless

    @property
    def _ix_after_dark_energy(self) -> int:
        return self._ix_after_massless + (2 if self.enable_dark_energy else 0)

    # --- Optional: massive neutrinos ------------------------------------------

    @property
    def has_massive_neutrinos(self) -> bool:
        return self.nqmax > 0

    def ix_psi(self, ell: int, q: int) -> int:
        """Return the state index of massive-neutrino multipole ``psi_l`` of bin ``q``.

        The bins follow the dark-energy block, bin-major and multipole-minor. That
        keeps each bin's free-streaming tail ``psi_3 ... psi_lmaxnu`` contiguous, one
        tridiagonal block of the sparsity pattern
        (:func:`discoeb.eb_model.perturbation_sparsity`).
        """
        if self.nqmax == 0:
            raise AttributeError("massive neutrinos are not enabled in this layout")
        if not (0 <= q < self.nqmax):
            raise IndexError(f"momentum bin q={q} out of range [0, {self.nqmax})")
        if not (0 <= ell <= self.lmaxnu):
            raise IndexError(f"multipole ell={ell} out of range [0, {self.lmaxnu}]")
        return self._ix_after_dark_energy + q * (self.lmaxnu + 1) + ell

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
        """Return the trimmed layout a cosmology's enabled components imply.

        ``w_DE_0 != -1`` or ``w_DE_a != 0`` turns on dynamical dark energy.
        ``num_massive_neutrinos > 0`` turns on massive neutrinos with ``nqmax``
        momentum bins. Curvature does not change the layout.
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


class _ParamCosmology(NamedTuple):
    """Hashable form of the DISCO-EB parameter dictionary."""

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


def _as_cosmology(cosmology, number=float):
    """Return a parameter dictionary as a ``_ParamCosmology``, anything else as is.

    ``number`` converts each value: ``float`` for the hashable form the caches
    key on, a JAX conversion to keep traced values.
    """

    if not isinstance(cosmology, dict):
        return cosmology
    return _ParamCosmology(
        Omegam=number(cosmology["Omegam"]),
        Omegab=number(cosmology["Omegab"]),
        w_DE_0=number(cosmology.get("w_DE_0", -1.0)),
        w_DE_a=number(cosmology.get("w_DE_a", 0.0)),
        cs2_DE=number(cosmology.get("cs2_DE", 1.0)),
        Omegak=number(cosmology.get("Omegak", 0.0)),
        A_s=number(cosmology["A_s"]),
        n_s=number(cosmology["n_s"]),
        H0=number(cosmology["H0"]),
        T_cmb=number(cosmology["Tcmb"]),
        Y_He=number(cosmology["YHe"]),
        Neff_massless=number(cosmology["Neff"]),
        num_massive_neutrinos=number(cosmology.get("Nmnu", 0.0)),
        mnu=number(cosmology.get("mnu", 0.0)),
        k_pivot=number(cosmology.get("k_p", 0.05)),
        tau_reio=number(cosmology.get("tau_reio", 0.0)),
    )


def _as_batch(cosmologies) -> tuple:
    """Return a cosmology or a sequence of them as a tuple (dictionaries unchanged)."""

    if isinstance(cosmologies, dict) or hasattr(cosmologies, "Omegam"):
        return (cosmologies,)
    return tuple(cosmologies)


# Solver and grid presets.
N_BACKGROUND_GRID = 4096
MATTER_THERMO_WEIGHT = 0.1
"""Thermal error weight (:data:`discoeb.recombination.THERMO_ERROR_WEIGHT`) of the
matter power spectrum solves. A recombination history ten times looser than
the hierarchy's tolerance moves ``P(k)`` by at most 6e-4 against the weight
of 1 the CMB solves keep for the visibility. It takes 150 driver steps per
trajectory instead of 225 (0.47 s against 0.62 s for 128 cosmologies x 128
modes on an RTX 4070 SUPER)."""
PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = 1.0e-2
PERTURB_MAX_STEPS = 20000
TAU_START = 0.1

# Columns of the packed parameter row the device right-hand side reads:
#   p = (grhog, grhornomass, grhoc, grhob, grhov,
#        tau_start, tau_end, k, cosmology_idx, save_index,
#        w_DE_0, w_DE_a, cs2_DE, grhok, grhor_nu, n_mnu, amnu,
#        YHe, Tcmb, H0, Omegab, z_re,
#        eps, dir_grhog, dir_grhornomass, dir_grhoc, dir_grhob, dir_grhov, dir_YHe, dir_Tcmb, dir_H0, dir_Omegab,
#        dir_z_re, dir_grhok, dir_w_DE_0, dir_w_DE_a, dir_cs2_DE, dir_grhor_nu, dir_n_mnu, dir_amnu)
# grhok = grhom * Omega_k is the constant comoving curvature density
# coefficient 8 pi G rho_K a^2, zero for a flat cosmology. The recombination
# rates and reionization read the five columns after ``amnu``
# (:data:`discoeb.recombination.THERMAL_PARAMETERS`). The model reads each input
# a derivative may move (:data:`DIRECTED_ROW_COLUMNS`: the five densities,
# ``YHe``, ``Tcmb``, ``H0``, ``Omegab``, ``z_re``, ``grhok``, the dark-energy
# and the massive-neutrino columns) as ``x_nominal + eps dir_x``. ``eps`` is zero in
# every primal solve, and the derivative in ``eps`` is the directional
# derivative along the row's ``dir`` vector, so each trajectory carries a
# single sensitivity column for its own direction (:func:`solve_eb`).
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
N_PARAM_HIERARCHY = IX_AMNU + 1
IX_YHE = N_PARAM_HIERARCHY
IX_TCMB = N_PARAM_HIERARCHY + 1
IX_H0 = N_PARAM_HIERARCHY + 2
IX_OMEGAB = N_PARAM_HIERARCHY + 3
IX_Z_RE = N_PARAM_HIERARCHY + 4
IX_EPS = N_PARAM_HIERARCHY + len(THERMAL_PARAMETERS)
IX_DIR = IX_EPS + 1
DIRECTED_NAMES = (
    "grhog",
    "grhornomass",
    "grhoc",
    "grhob",
    "grhov",
    "YHe",
    "Tcmb",
    "H0",
    "Omegab",
)
"""The inputs a trajectory's sensitivity direction may move, in the order of :data:`DIRECTED_COLUMNS`."""
DIRECTED_COLUMNS = (0, 1, 2, 3, 4, IX_YHE, IX_TCMB, IX_H0, IX_OMEGAB)
"""The columns of :data:`DIRECTED_NAMES`: their nominal values, with ``eps`` and the ``dir`` block after the row."""
DIRECTED_ROW_NAMES = (
    *DIRECTED_NAMES,
    "z_re",
    "grhok",
    "w_DE_0",
    "w_DE_a",
    "cs2_DE",
    "grhor_nu",
    "n_mnu",
    "amnu",
)
"""Every row input a direction may move: :data:`DIRECTED_NAMES`, the inputs a
caller may give, then the reionization redshift, the curvature, the
dark-energy and the massive-neutrino inputs, which :func:`solve_eb` moves
with the parameters they come from."""
DIRECTED_ROW_COLUMNS = (
    *DIRECTED_COLUMNS,
    IX_Z_RE,
    IX_GRHOK,
    IX_W_DE_0,
    IX_W_DE_A,
    IX_CS2_DE,
    IX_GRHOR_NU,
    IX_NMNU,
    IX_AMNU,
)
"""The columns of :data:`DIRECTED_ROW_NAMES`, whose directions fill the ``dir`` block."""
N_PARAM = IX_DIR + len(DIRECTED_ROW_COLUMNS)
IX_DY0 = N_PARAM
"""Where the derivative launch's row continues past the plain one: the
initial state's tangent ``dy0`` (``nvar`` columns), then the time flag
(:func:`build_keep_hook`)."""

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
    "w_DE_0_nominal": IX_W_DE_0,
    "w_DE_a_nominal": IX_W_DE_A,
    "cs2_DE_nominal": IX_CS2_DE,
    "grhok_nominal": IX_GRHOK,
    "grhor_nu_nominal": IX_GRHOR_NU,
    "n_mnu_nominal": IX_NMNU,
    "amnu_nominal": IX_AMNU,
    **{name: N_PARAM_HIERARCHY + i for i, name in enumerate(THERMAL_PARAMETERS)},
    "eps": IX_EPS,
    **{f"dir_{name}": IX_DIR + i for i, name in enumerate(DIRECTED_ROW_NAMES)},
}
"""The row's columns by the names the component model's providers take them under
(:mod:`discoeb.eb_model`, :mod:`discoeb.recombination`)."""

TRAJECTORIES_PER_BLOCK = 32
"""Wave modes packed into one CUDA block by modax's thread-local Rodas5P kernel."""

CUDA_MAX_REGISTERS = int(os.environ.get("DISCO_CUDA_MAX_REGISTERS", "168"))
"""Per-thread register cap for that kernel: fewer registers, more resident warps."""


BATCH_CACHE_SIZE = 8
"""Batches whose prepared solve and packed initial states are kept."""


def _densities(cosmology) -> dict:
    """Return :func:`discoeb.eb_model.density_parameters` of a cosmology as floats, memoised."""

    return _densities_of(_as_cosmology(cosmology))


@functools.lru_cache(maxsize=4096)
def _densities_of(cosmology: _ParamCosmology) -> dict:
    d = density_parameters(cosmology.to_background_params(), xp=np)
    return {k: float(v) for k, v in d.items()}


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """Return ``(grhog, grhornomass, grhoc, grhob, grhov)`` of :func:`discoeb.eb_model.density_parameters`."""

    d = _densities(cosmology)
    return (d["grhog"], d["grhornomass"], d["grhoc"], d["grhob"], d["grhov"])


# ============================================================
# THE JOINT MODEL AND ITS KERNEL
# ============================================================


def _layout_for(cosmology) -> PerturbationLayout:
    """Return a cosmology's perturbation layout at the components' current truncations."""

    return PerturbationLayout.from_cosmology(
        cosmology,
        lmaxg=Photons.LMAX_G,
        lmaxpol=Photons.LMAX_POL,
        lmaxr=MasslessNeutrinos.LMAX_NR,
        lmaxnu=MassiveNeutrinos.LMAX_NU,
        nqmax=MassiveNeutrinos.NQMAX,
    )


def joint_model(layout, extra=()):
    """Return the hierarchy of ``layout`` with :class:`~discoeb.recombination.Recombination` appended.

    The recombination states come after the hierarchy's, so the
    :class:`PerturbationLayout` indices still hold. The five thermal states
    (``ln a``, ``x_H``, ``x_He``, ``T_m`` and the optical depth) are the last
    columns of every history, at ``model.index["lna"]`` and so on. ``extra``
    lists component classes to append after them (the line-of-sight sources).
    """

    return perturbation_model(
        layout, extra=[Recombination(), *(cls() for cls in extra)]
    )


def thermal_indices(model) -> tuple[int, ...]:
    """Return the flat indices of the five recombination states in ``model``."""

    return tuple(
        model.index[n] for n in ("lna", "thermo_0", "thermo_1", "thermo_2", "kappa")
    )


THERMAL_STATES = ("lna", "thermo", "kappa")
"""The recombination block's states, in model order."""

THERMAL_QUANTITIES = ("a", "opacity", "cs2", "rhonu")
"""What the hierarchy reads of the recombination block: the scale factor, the
opacity, the baryon sound speed and the massive-neutrino density ratio. The
block publishes them and modax interpolates them between the block's steps."""


def _restrict(pattern: np.ndarray, indices) -> np.ndarray:
    """Return the ``(nnz, 2)`` pattern's rows and columns ``indices``, renumbered."""

    position = {idx: k for k, idx in enumerate(indices)}
    kept = [
        (position[r], position[c])
        for r, c in pattern.tolist()
        if r in position and c in position
    ]
    return np.array(sorted(kept), dtype=np.int64).reshape(-1, 2)


def _shifted(fn, indices, reads_blocks: bool = False):
    """Return ``fn(u, tau, p[, q])`` reading its block of the state as ``y = u + eps dy0``.

    ``fn`` is a device function of the block's packed state ``y``. ``indices``
    are the block's model indices, so ``dy0`` comes in model order from the
    row's :data:`IX_DY0` columns. modax starts ``u``'s sensitivity at zero, so
    ``y``'s starts at ``dy0``, the initial state's tangent. With ``eps = 0``
    it is ``fn`` itself.
    """

    tail = ", q" if reads_blocks else ""
    state = ", ".join(
        f"u[{i}] + p[{IX_EPS}] * p[{IX_DY0 + m}]" for i, m in enumerate(indices)
    )
    source = f"def shifted(u, tau, p{tail}):\n    return fn(({state},), tau, p{tail})\n"
    return _compile("shifted", source, {"fn": fn})


@functools.lru_cache(maxsize=None)
def joint_kernel(layout, extra=(), shift=False):
    """Return ``(model, hierarchy)`` of the joint model, cached per layout and ``extra``.

    The hierarchy is a modax :class:`~modax.Hierarchy` of two blocks. The
    recombination block (:data:`THERMAL_STATES`) publishes
    :data:`THERMAL_QUANTITIES`, and the perturbation block's right-hand side
    takes them as given. modax steps the recombination block ahead of the
    perturbations and hands the quantities down as a quintic Hermite
    interpolant. So the perturbation Jacobian has no thermal columns, and the
    thermal providers run once per recombination step. The state keeps the
    joint model's order, thermal states last.

    With ``shift`` the blocks read their states as ``y = u + eps dy0``
    (:func:`_shifted`), for the derivative launch of :func:`solve_eb`, whose
    rows carry the initial state's tangent ``dy0``. The pattern is the same.
    """

    from modax import Block, Hierarchy
    from numba_cuda_mlir import cuda

    model = joint_model(layout, extra)
    pattern = model.sparsity()
    thermal = model.state_indices(THERMAL_STATES)
    others = [name for name in model.states if name not in THERMAL_STATES]
    hierarchy_idx = model.state_indices(others)

    def read(fn, indices, reads_blocks=False, device=False):
        if not shift:
            return fn
        return _shifted(
            fn if device else cuda.jit(device=True)(fn), indices, reads_blocks
        )

    recombination = Block(
        read(
            model.build(PARAM_COLUMNS, states=THERMAL_STATES, name="recombination_rhs"),
            thermal,
        ),
        len(thermal),
        outputs=read(
            model.build_function(
                THERMAL_QUANTITIES,
                inputs=("y", "tau"),
                params=PARAM_COLUMNS,
                states=THERMAL_STATES,
                name="thermal_quantities",
            ),
            thermal,
            device=True,
        ),
        n_outputs=len(THERMAL_QUANTITIES),
        sparsity=_restrict(pattern, thermal),
        name="recombination",
    )
    hierarchy = Block(
        read(
            model.build(
                PARAM_COLUMNS,
                states=others,
                quantities=THERMAL_QUANTITIES,
                name="hierarchy_rhs",
            ),
            hierarchy_idx,
            reads_blocks=True,
        ),
        len(hierarchy_idx),
        reads=(recombination,),
        sparsity=_restrict(pattern, hierarchy_idx),
        name="hierarchy",
    )
    return model, Hierarchy(hierarchy, layout=(hierarchy, recombination))


def _stack_background_params(cosmologies) -> dict:
    """Stack the cosmologies' background parameters by name, one vmap lane each."""

    rows = [c.to_background_params() for c in cosmologies]
    return {
        key: jnp.asarray([row[key] for row in rows], dtype=jnp.float64)
        for key in rows[0]
    }


def _background_tau(d):
    """Return ``(a_grid, tau_grid)`` of the conformal-time quadrature for densities ``d`` (JAX)."""

    a_bg = jnp.geomspace(1.0e-9, 1.0, N_BACKGROUND_GRID)
    dt = jax.vmap(dtauda, in_axes=(0, None))(a_bg, d)
    steps = 0.5 * (dt[1:] + dt[:-1]) * jnp.diff(a_bg)
    radiation = (
        d["grhog"]
        + d["grhornomass"]
        + d["grhor_nu"] * d["n_mnu"] * neutrino_density_ratio(1.0e-9, d["amnu"])
    )
    taumin = 1.0e-9 / jnp.sqrt(radiation / 3.0)
    return a_bg, jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(steps))) + taumin


def _reionization_redshift(a_bg, tau_bg, params, tau_reio):
    """Bisect for ``z_re`` of CAMB's step on the background grid.

    Returns 0 unless ``tau_reio > 0``. Neglects the residual ionization left by
    recombination (``xe_rec = 0``), which shifts ``tau_reio`` by 2e-4 of itself.
    """

    dtau = jnp.gradient(tau_bg)
    thermal = {name: params[name] for name in ("YHe", "Omegab", "H0")}

    def tau_of(z_re):
        xe_reio, akthom = recombination_model().evaluate(
            "xe_reio",
            "akthom",
            math=jnp,
            inv_a=1.0 / a_bg,
            xe_rec=0.0,
            z_re=z_re,
            **thermal,
        )
        return jnp.sum(xe_reio * akthom / a_bg**2 * dtau)

    def bisect(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        too_much = tau_of(mid) > tau_reio
        return (jnp.where(too_much, lo, mid), jnp.where(too_much, mid, hi)), None

    (lo, hi), _ = jax.lax.scan(
        bisect, (jnp.asarray(1.0), jnp.asarray(40.0)), None, length=40
    )
    z_re = 0.5 * (lo + hi)
    # The bisection's midpoint carries no derivative. The implicit function
    # theorem gives it, d z_re = -d(tau_of - tau_reio) / (d tau_of / d z_re),
    # here as a term that is zero in value.
    residual = tau_of(z_re) - tau_reio
    slope = jax.lax.stop_gradient(jax.grad(tau_of)(z_re))
    slope = jnp.where(tau_reio > 0.0, slope, 1.0)
    z_re = z_re - (residual - jax.lax.stop_gradient(residual)) / slope
    return jnp.where(tau_reio > 0.0, z_re, 0.0)


@jax.jit
def background_endpoints(stacked):
    """Return ``(tau0, lna_start, z_re)`` of stacked background parameters, one vmap lane each.

    ``tau0`` is the conformal age, where each trajectory stops. ``lna_start`` is
    ``ln a`` at :data:`TAU_START`, where it starts. ``z_re`` is the reionization
    redshift that gives ``tau_reio`` (0 without reionization). The bisection for
    it runs on the background and neglects the residual ionization left by
    recombination, which shifts ``tau_reio`` by 2e-4 of itself.
    """

    def one(params):
        a_bg, tau_bg = _background_tau(density_parameters(params))
        lna_start = jnp.log(jnp.interp(TAU_START, tau_bg, a_bg))
        z_re = _reionization_redshift(a_bg, tau_bg, params, params["tau_reio"])
        return tau_bg[-1], lna_start, z_re

    return jax.vmap(one)(stacked)


@jax.jit
def _conformal_times(stacked, a):
    """Return each cosmology's conformal times of the scale factors ``a``, ``(n_cosmologies, len(a))``.

    Linear in ``ln a`` on the background quadrature's grid, whose last point
    is ``a = 1``, so ``a = 1`` gives ``tau0`` exactly.
    """

    def one(params):
        a_bg, tau_bg = _background_tau(density_parameters(params))
        return jnp.interp(jnp.log(a), jnp.log(a_bg), tau_bg)

    return jax.vmap(one)(stacked)


A_MIN = 1.0e-6
"""The smallest scale factor :func:`solve_eb` saves at: a few times ``a`` at
:data:`TAU_START`, where every trajectory starts."""


class _PreparedSolve(NamedTuple):
    """What a batch shares: the compiled joint model and each cosmology's endpoints."""

    cosmologies: tuple
    layout: PerturbationLayout
    model: Model
    ode_fn: object  # the modax Hierarchy
    tau0: np.ndarray  # (n_cosmologies,)
    lna_start: np.ndarray
    z_re: np.ndarray


def _prepare_solve(cosmologies, extra=()) -> _PreparedSolve:
    """Return the joint kernel and endpoints of a batch of cosmologies, cached per batch.

    All cosmologies in a batch must share a perturbation layout. Nothing here
    depends on the wave modes, save times or tolerances. The kernel compiles
    once per layout, and the endpoints cost a vmapped quadrature of a few
    milliseconds.
    """

    return _prepared(
        tuple(_as_cosmology(cosmology) for cosmology in cosmologies), extra
    )


@functools.lru_cache(maxsize=BATCH_CACHE_SIZE)
def _prepared(cosmologies, extra) -> _PreparedSolve:
    layouts = {_layout_for(cosmology) for cosmology in cosmologies}
    if len(layouts) > 1:
        raise ValueError(
            "all cosmologies in a batch must share a perturbation layout "
            "(same dark-energy / massive-neutrino settings)"
        )
    (layout,) = layouts
    model, hierarchy = joint_kernel(layout, extra)
    tau0, lna_start, z_re = (
        np.asarray(x)
        for x in background_endpoints(_stack_background_params(cosmologies))
    )
    return _PreparedSolve(cosmologies, layout, model, hierarchy, tau0, lna_start, z_re)


def _prepare_traced(cosmologies, layout) -> _PreparedSolve:
    """Return the prepared solve of cosmologies of JAX values, as under ``jit``.

    The endpoints stay JAX values, so nothing is cached and :func:`_pack_batch`
    packs the batch in JAX. The layout is given, since the values that choose
    it may be traced.
    """

    model, hierarchy = joint_kernel(layout)
    tau0, lna_start, z_re = background_endpoints(_stack_background_params(cosmologies))
    return _PreparedSolve(cosmologies, layout, model, hierarchy, tau0, lna_start, z_re)


def _row(cosmology, tau_end: float, z_re: float, xp=np) -> dict:
    """Return one cosmology's parameter row by name, without the wave number.

    ``xp`` is ``numpy`` for a cosmology of floats, ``jax.numpy`` for one of
    JAX values, whose densities are then traced.
    """

    if xp is np:
        d = _densities(cosmology)
    else:
        d = density_parameters(cosmology.to_background_params(), xp=xp)
    return {
        **{
            f"{name}_nominal": d[name]
            for name in ("grhog", "grhornomass", "grhoc", "grhob", "grhov")
        },
        **{
            f"{name}_nominal": d[name]
            for name in ("grhok", "grhor_nu", "n_mnu", "amnu")
        },
        "tau_start": TAU_START,
        "tau_end": tau_end,
        "save_index": 1.0,
        "w_DE_0_nominal": cosmology.w_DE_0,
        "w_DE_a_nominal": cosmology.w_DE_a,
        "cs2_DE_nominal": cosmology.cs2_DE,
        "YHe_nominal": cosmology.Y_He,
        "Tcmb_nominal": cosmology.T_cmb,
        "H0_nominal": cosmology.H0,
        "Omegab_nominal": cosmology.Omegab,
        "z_re_nominal": z_re,
    }


def _row_vector(cosmology, tau_end: float, z_re: float, xp=np) -> np.ndarray:
    """Return :func:`_row` as the ``N_PARAM`` columns of :data:`PARAM_COLUMNS`, zero where it has no value."""

    row = _row(cosmology, tau_end, z_re, xp)
    if xp is not np:
        values = {column: row.get(name, 0.0) for name, column in PARAM_COLUMNS.items()}
        return xp.stack(
            [xp.asarray(values.get(c, 0.0), dtype=xp.float64) for c in range(N_PARAM)]
        )
    out = np.zeros(N_PARAM, dtype=np.float64)
    for name, column in PARAM_COLUMNS.items():
        out[column] = row.get(name, 0.0)
    return out


def _initial_inputs(cosmologies, spread, xp=np) -> dict:
    """Return what :meth:`Model.initial_state` reads of the cosmologies.

    ``spread`` maps one value per cosmology to the trajectories' values.
    """

    if xp is np:
        densities = np.array([density_coefficients(c) for c in cosmologies]).T
    else:
        traced = [
            density_parameters(c.to_background_params(), xp=xp) for c in cosmologies
        ]
        densities = [xp.stack([d[name] for d in traced]) for name in DIRECTED_NAMES[:5]]
    return {
        **dict(zip(DIRECTED_NAMES[:5], map(spread, densities))),
        "w_DE_0": spread([c.w_DE_0 for c in cosmologies]),
        "w_DE_a": spread([c.w_DE_a for c in cosmologies]),
        "Tcmb": spread([c.T_cmb for c in cosmologies]),
    }


def make_params(
    cosmology, k_values: np.ndarray, tau_end: float, z_re: float = 0.0
) -> np.ndarray:
    """Return packed parameters for the device right-hand side, one row per wave mode."""

    params = np.tile(
        _row_vector(_as_cosmology(cosmology), tau_end, z_re), (len(k_values), 1)
    )
    params[:, IX_K] = k_values
    return params


def make_initial_states(
    cosmology, k_values: np.ndarray, model, lna_start
) -> np.ndarray:
    """Return adiabatic initial states plus the fully ionized thermal state, ``(n_k, nvar)``.

    ``cosmology`` is one cosmology or a sequence of ``len(k_values)``, one per
    trajectory. ``lna_start`` is a number or one per trajectory. The initial
    conditions broadcast over the trajectories.
    """

    cosmologies = [_as_cosmology(c) for c in _as_batch(cosmology)]

    def per(values):
        return values[0] if len(values) == 1 else np.asarray(values)

    return model.initial_state(
        k_values,
        tau=TAU_START,
        lna_start=np.asarray(lna_start, dtype=np.float64),
        **_initial_inputs(cosmologies, per),
    )


@functools.lru_cache(maxsize=BATCH_CACHE_SIZE)
def _packed(cosmologies, model, tau0, lna_start, z_re, k_bytes, tau_bytes):
    k_arr = np.frombuffer(k_bytes, dtype=np.float64)
    if tau_bytes is None:
        t_span = np.asarray((TAU_START, float(np.max(tau0))), dtype=np.float64)
    else:
        t_span = np.frombuffer(tau_bytes, dtype=np.float64).copy()
    params, y0, order, k_sorted = _pack(
        cosmologies, model, tau0, lna_start, z_re, k_arr
    )
    return (
        np.ascontiguousarray(params),
        np.ascontiguousarray(y0),
        t_span,
        order,
        k_sorted,
    )


def _pack(cosmologies, model, tau0, lna_start, z_re, k_arr, xp=np):
    """Return ``(params, y0, order, k_sorted)``: the rows and initial states of a batch.

    ``xp`` is ``numpy`` for cosmologies of floats and ``jax.numpy`` for ones
    of JAX values, whose rows and initial states are then traced.
    """

    n_cosmo = len(cosmologies)
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]
    rows = xp.stack(
        [_row_vector(c, t, z, xp) for c, t, z in zip(cosmologies, tau0, z_re)]
    )
    params = xp.tile(rows, (len(k_sorted), 1))
    k_column = np.repeat(k_sorted, n_cosmo)
    tags = np.tile(np.arange(n_cosmo, dtype=np.float64), len(k_sorted))
    if xp is np:
        params[:, IX_K] = k_column
        params[:, IX_COSMOLOGY] = tags
    else:
        params = (
            jnp.asarray(params).at[:, IX_K].set(k_column).at[:, IX_COSMOLOGY].set(tags)
        )
    # per-cosmology inputs, tiled over the wave modes (row = k_idx * n_cosmo + c)

    def tile(values):
        return xp.tile(xp.asarray(values, dtype=xp.float64), len(k_sorted))

    y0 = model.initial_state(
        k_column,
        xp=xp,
        tau=TAU_START,
        lna_start=tile(lna_start),
        **_initial_inputs(cosmologies, tile, xp),
    )
    return params, y0, order, k_sorted


def _pack_batch(prepared, k_values, tau_save=None):
    """Return (and cache) the packed ``(params, y0, t_span, order, k_sorted)`` of a batch.

    Rows go as ``row = k_idx * n_cosmo + cosmology_idx``, so consecutive rows
    share a wave mode. ``tau_save`` is the shared grid of save times (default:
    the two endpoints). :func:`_launch` clips it to each trajectory's own
    ``tau0``.
    """

    if not isinstance(prepared.tau0, np.ndarray):
        # a batch of JAX values (traced under jit): packed in JAX, not cached
        params, y0, order, k_sorted = _pack(
            prepared.cosmologies,
            prepared.model,
            prepared.tau0,
            prepared.lna_start,
            prepared.z_re,
            np.asarray(k_values, dtype=np.float64),
            xp=jnp,
        )
        t_span = (
            jnp.stack([jnp.asarray(TAU_START), jnp.max(prepared.tau0)])
            if tau_save is None
            else np.asarray(tau_save, dtype=np.float64)
        )
        return params, y0, t_span, order, k_sorted
    k_bytes = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64)).tobytes()
    tau_bytes = (
        None
        if tau_save is None
        else np.ascontiguousarray(np.asarray(tau_save, dtype=np.float64)).tobytes()
    )
    return _packed(
        prepared.cosmologies,
        prepared.model,
        # one host copy of each array, not a device read per element
        tuple(np.asarray(prepared.tau0, dtype=np.float64).tolist()),
        tuple(np.asarray(prepared.lna_start, dtype=np.float64).tolist()),
        tuple(np.asarray(prepared.z_re, dtype=np.float64).tolist()),
        k_bytes,
        tau_bytes,
    )


def error_weights_for(
    prepared, n_traj: int, base=None, thermo_weight: float = THERMO_ERROR_WEIGHT
):
    """Return the solver's error weights ``(n_traj, nvar)``.

    ``base`` (or ones) weights the hierarchy and ``thermo_weight`` the
    recombination block. The optical depth gets the block's weight too, so its
    error counts against ``rtol |kappa|``. The block's steps then resolve its
    increments where ``kappa`` is of order one, through recombination and after,
    which is where ``exp(-kappa)`` reads them. Unweighted, it left the lensed TT
    0.5 percent off CLASS at ``l = 1500``.
    """

    model = prepared.model
    weights = np.ones((n_traj, model.nvar), dtype=np.float64)
    if base is not None:
        base = np.asarray(base, dtype=np.float64)
        weights[:, : base.shape[-1]] = base
    weights[:, list(thermal_indices(model))] = thermo_weight
    return weights


def _launch(
    prepared,
    params,
    y0,
    t_span,
    *,
    rtol,
    atol,
    first_step,
    max_steps,
    error_weights=None,
    thermo_weight=THERMO_ERROR_WEIGHT,
    lu_precision="fp32",
    max_registers=CUDA_MAX_REGISTERS,
    **extra,
):
    """Run one modax Rodas5P launch of the prepared hierarchy over the packed trajectories.

    The method object carries the numerics: the hierarchy's LU precision and the
    PI controller's gains. The recombination block, which the hierarchy reads,
    always factorises in double. Each trajectory gets its own row of save
    times, the shared ``t_span`` clipped to its ``tau0`` in ``IX_TAU_END``, so
    it stops there and its later saves hold its final state.
    """

    from modax import PIDGains, Rodas5P
    from modax import solve as modax_solve

    method = Rodas5P(lu_precision=lu_precision, gains=PIDGains(pcoeff=0.3, icoeff=0.4))
    if isinstance(params, np.ndarray):
        t_rows = np.minimum(np.atleast_2d(t_span), params[:, IX_TAU_END, None])
    else:
        tau_end = jax.lax.stop_gradient(params[:, IX_TAU_END, None])
        t_rows = jnp.minimum(jnp.atleast_2d(t_span), tau_end)
    return modax_solve(
        prepared.ode_fn,
        y0,
        t_rows,
        params,
        method=method,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        max_registers=max_registers,
        error_weights=error_weights_for(
            prepared, np.shape(params)[0], error_weights, thermo_weight
        ),
        cache=True,
        **extra,
    )


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

    One modax launch integrates all ``n_cosmologies * n_k`` trajectories, each
    with its recombination block stepping ahead of its hierarchy
    (:func:`joint_kernel`). Each trajectory stops at its own ``tau0``, and later
    save times hold its final state. Returns ``(history, k_sorted, prepared)``,
    with ``history`` a device array ``(n_k, n_cosmologies, len(tau_save),
    nvar)`` in ascending ``k``.
    """

    prepared = _prepare_solve(tuple(cosmologies))
    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_save)
    sol = _launch(
        prepared,
        params,
        y0,
        t_span,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )
    hist = jnp.asarray(sol).reshape(
        len(k_sorted), len(prepared.cosmologies), len(t_span), prepared.model.nvar
    )
    return hist, k_sorted, prepared


def _primordial_power(ks, A_s, n_s, k_pivot):
    """``2 pi^2 / k^3`` times the primordial curvature spectrum: ``P(k) = this * delta_m^2``."""

    return 2.0 * np.pi**2 / ks**3 * A_s * (ks / k_pivot) ** (n_s - 1.0)


def _power_spectrum_from_final(y_final, k_sorted, cosmologies, k_values):
    """Return ``P(k)`` ``(n_cosmo, n_k)`` in the caller's order from final states ``(n_k, n_cosmo, nvar)``."""

    dens = np.array([density_coefficients(c) for c in cosmologies])
    grhoc, grhob = dens[:, 2][None, :], dens[:, 3][None, :]
    delta_m = (grhoc * y_final[:, :, IX_CLXC] + grhob * y_final[:, :, IX_CLXB]) / (
        grhoc + grhob
    )
    pk_sorted = (
        _primordial_power(
            k_sorted[:, None],
            *(
                np.array([getattr(c, a) for c in cosmologies])[None, :]
                for a in ("A_s", "n_s", "k_pivot")
            ),
        )
        * delta_m**2
    )
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
    thermo_weight: float = MATTER_THERMO_WEIGHT,
) -> np.ndarray:
    """Return ``P(k)`` in Mpc^3, shape ``(n_cosmologies, n_k)``, for a batch in one launch."""

    prepared = _prepare_solve(tuple(cosmologies))
    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values)
    sol = _launch(
        prepared,
        params,
        y0,
        t_span,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )
    y_final = np.asarray(sol)[:, -1, :].reshape(
        len(k_sorted), len(prepared.cosmologies), prepared.model.nvar
    )
    return _power_spectrum_from_final(y_final, k_sorted, prepared.cosmologies, k_values)


# ============================================================
# DIFFERENTIABLE SOLVES
# ============================================================


def _tau0_of(densities, d):
    """Return the conformal age as a JAX function of the five densities, the rest of the background from ``d``."""

    return _background_tau({**d, **dict(zip(DIRECTED_NAMES[:5], densities))})[1][-1]


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
    thermo_weight: float = MATTER_THERMO_WEIGHT,
):
    """Return ``P(k)`` as a JAX array, differentiable in the densities and thermal inputs.

    ``densities`` (the five of :data:`DIRECTED_NAMES`) and ``thermal`` (``YHe,
    Tcmb, H0, Omegab``) default to the cosmology's own. The derivative is the
    total one. The densities enter the hierarchy, the recombination rates
    (through the expansion rate), the matter weighting of ``delta_m`` and the
    conformal age at which the solve reads the state. The last path goes through
    ``y(tau_0(d)) = y(tau_0^*) + y'(tau_0^*) (tau_0(d) - tau_0^*)``, which is
    zero in value and carries ``d tau_0 / d(densities)`` from the background
    quadrature. The solve reads the value at the conformal age of the given
    densities, so a finite difference sees the same end-time dependence. Only
    the initial state stays fixed.

    The solve is the hooked launch of :func:`_kept_values`, keeping the
    contrasts and the rates the end-time term needs at ``tau_0``.
    """

    cosmology = _as_cosmology(cosmology)
    prepared = _prepare_solve((cosmology,))
    d = _densities(cosmology)
    if densities is None:
        densities = [d[name] for name in DIRECTED_NAMES[:5]]
    if thermal is None:
        thermal = [cosmology.Y_He, cosmology.T_cmb, cosmology.H0, cosmology.Omegab]
    x = jnp.concatenate(
        [
            jnp.asarray(densities, dtype=jnp.float64),
            jnp.asarray(thermal, dtype=jnp.float64),
        ]
    )
    kept = _kept_values(
        prepared,
        k_values,
        # a save past tau0 holds the state today
        np.array([TAU_START, 2.0 * float(prepared.tau0[0])]),
        ("clxc", "clxb", "z", "vb"),
        tau_end_of=lambda xs: jnp.stack([_tau0_of(xi[:5], d) for xi in xs]),
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )
    clxc, clxb, z, vb = kept(x[None, :])[0, :, :, 0]

    # End-time term: the state's rate at tau0 (the equations the hook
    # evaluates on the final state) times d tau0 / d densities.
    ks = jnp.asarray(np.asarray(k_values, dtype=np.float64))
    tau0 = _tau0_of(x[:5], d)
    dtau = tau0 - jax.lax.stop_gradient(tau0)
    clxc = clxc - ks * z * dtau
    clxb = clxb - ks * (z + vb) * dtau

    grhoc, grhob = x[2], x[3]
    delta_m = (grhoc * clxc + grhob * clxb) / (grhoc + grhob)
    return (
        _primordial_power(ks, cosmology.A_s, cosmology.n_s, cosmology.k_pivot)
        * delta_m**2
    )


def background_history(cosmology):
    """Return ``(a, tau)`` of one cosmology's background quadrature, host arrays ending at ``tau0``."""

    return _background_history(_as_cosmology(cosmology))


@functools.lru_cache(maxsize=64)
def _background_history(cosmology: _ParamCosmology):
    d = {k: jnp.asarray(v) for k, v in _densities(cosmology).items()}
    return tuple(np.asarray(x) for x in _background_tau(d))


# =============================================================================
# solve_eb: keep only the selected quantities, differentiable in the cosmology
# =============================================================================


@dataclass(frozen=True)
class EBSolution:
    """What :func:`solve_eb` kept: the selected quantities on the requested grid.

    ``values`` is a device array ``(n_cosmologies, len(names), n_k, n_tau)``
    with wave modes and times in the caller's order. ``solution[name]`` is that
    quantity's ``(n_cosmologies, n_k, n_tau)`` block. ``tau0`` is each
    cosmology's conformal age, past which its column holds the state today.
    For a solve at scale factors, ``a`` holds them and ``tau`` each
    cosmology's conformal times of them, ``(n_cosmologies, len(a))``.
    ``values`` is differentiable in the parameters of
    :data:`DIFFERENTIABLE_PARAMETERS` given as JAX values.
    """

    names: tuple[str, ...]
    k: np.ndarray
    tau: np.ndarray
    tau0: np.ndarray
    values: jax.Array
    layout: PerturbationLayout
    a: np.ndarray | None = None

    def __getitem__(self, name: str) -> jax.Array:
        if name not in self.names:
            raise KeyError(f"{name!r} was not kept; the solution holds {self.names}")
        return self.values[:, self.names.index(name)]


# a pytree, so a jitted function may return it
jax.tree_util.register_dataclass(
    EBSolution,
    data_fields=["k", "tau", "tau0", "values", "a"],
    meta_fields=["names", "layout"],
)


def eps_sensitivity(nvar: int, n_params: int = N_PARAM):
    """Return the sensitivity spec of a joint launch along the parameter row's ``eps``.

    It adds one column ``S = dy / d eps`` after the state, outside step control.
    The model reads each directed input as ``x_nominal + eps dir_x``
    (:data:`DIRECTED_COLUMNS`), so ``S`` is the derivative along the row's
    ``dir``, through both the hierarchy and the recombination block.
    ``n_params`` is the row's width (wider in :func:`solve_eb`'s derivative
    launch).
    """

    from modax.rodas5P import SensitivitySpec

    return SensitivitySpec(
        nvar,
        n_params,
        wrt_y0=False,
        wrt_params=True,
        error_control=False,
        param_columns=(IX_EPS,),
    )


def build_keep_hook(model, names, *, tangent: bool = False, shifted: bool = False):
    """Return a modax save hook that stores the quantities ``names`` at every save.

    A name is a state element or a quantity the joint model provides. State
    elements (``clxc``, ``theta_2``, ``psi_0_1``: the
    :func:`~discoeb.eb_model.layout_state_names`, or ``lna``, ``thermo_0`` ...
    ``kappa`` of the recombination block) come straight from the state. Provided
    quantities (``R``, ``sigma``, ``phi`` with the line-of-sight sources
    appended, ``opacity`` or ``cs2`` from the thermal states) come from
    :meth:`~discoeb.component.Model.build_function`, with the state, the time and
    the parameter row as inputs. The hook clamps the time to the trajectory's
    own ``tau_end``: a save past a cosmology's ``tau0`` holds its final state,
    and the background at ``tau0`` goes with it. ``hook._generated_source``
    holds the generated device function of ``(y, tau, p)``, which modax's
    :func:`~modax.rodas5P.quantity_hook` turns into the hook.

    With ``tangent``, the hook serves a joint launch that carries one
    sensitivity column ``S = dy/d eps`` after the state
    (:func:`eps_sensitivity`). It stores each quantity and, after the save's
    quantities, each one's total derivative along the column, ``dq/d eps =
    (dq/dy) S + dq/d eps``. modax forms these in one Enzyme forward sweep of the
    quantities seeded with ``(S, 0, e_eps)``, built for the kernel's own target.

    With ``shifted``, the hook serves :func:`solve_eb`'s derivative launch, on
    :func:`joint_kernel`'s shifted blocks. Its row carries the initial state's
    tangent ``dy0`` at :data:`IX_DY0` and a time flag after it. The hook reads
    the state as the blocks do, ``y = u + eps dy0``, and moves the time by
    ``dt = eps * flag`` and the state with it by ``dt f``, ``f`` the joint
    model's right-hand side. A replica with the flag set and no direction
    stores each quantity's rate ``dq/dtau`` as its tangent.
    """

    from modax.rodas5P import quantity_hook
    from numba_cuda_mlir import cuda

    names = tuple(names)
    if not names:
        raise ValueError("keep at least one quantity")
    vectors = [n for n in names if n in model.states and n not in model.index]
    if vectors:
        raise ValueError(
            f"{vectors} are vector states: name their elements ({', '.join(model.states[vectors[0]].elements[:3])}, ...)"
        )
    provided = [n for n in names if n not in model.index]
    namespace: dict = {}
    lines = [
        "def quantities(y, tau, p):",
        f"    t = p[{IX_TAU_END}]",
        "    if tau < t:",
        "        t = tau",
    ]
    n_params = N_PARAM
    if shifted:
        nvar = model.nvar
        n_params = IX_DY0 + nvar + 1
        namespace["rhs"] = cuda.jit(device=True)(
            model.build(PARAM_COLUMNS, name="joint_rhs")
        )
        state = ", ".join(f"u[{i}] + e * p[{IX_DY0 + i}]" for i in range(nvar))
        advanced = ", ".join(f"y[{i}] + dt * f[{i}]" for i in range(nvar))
        lines[0] = "def quantities(u, tau, p):"
        lines += [
            f"    e = p[{IX_EPS}]",
            f"    y = ({state},)",
            f"    dt = e * p[{IX_DY0 + nvar}]",
            "    f = rhs(y, t, p)",
            f"    y = ({advanced},)",
            "    t = t + dt",
        ]
    if provided:
        # build_function returns a lone output bare, so pair it with tau.
        outputs = provided if len(provided) > 1 else [*provided, "tau"]
        namespace["kept"] = model.build_function(
            outputs, inputs=("y", "tau"), params=PARAM_COLUMNS, name="kept"
        )
        targets = ", ".join(f"q_{i}" for i in range(len(outputs)))
        lines.append(f"    {targets} = kept(y, t, p)")
    values = [
        f"y[{model.index[name]}]"
        if name in model.index
        else f"q_{provided.index(name)}"
        for name in names
    ]
    lines.append(f"    return ({', '.join(values)},)")
    quantities = _compile("quantities", "\n".join(lines) + "\n", namespace)
    source = quantities._generated_source
    hook = quantity_hook(
        quantities,
        len(names),
        model.nvar,
        n_params,
        eps_sensitivity(model.nvar, n_params) if tangent else None,
    )
    setattr(hook, "_generated_source", source)
    return hook


@functools.lru_cache(maxsize=None)
def _keep_hook(layout, extra, names, tangent, shifted=False):
    """Return the compiled hook for a layout, extra components, names and flags (cached)."""

    return build_keep_hook(
        joint_model(layout, extra), names, tangent=tangent, shifted=shifted
    )


DIFFERENTIABLE_PARAMETERS = (
    "Omegam",
    "Omegab",
    "H0",
    "Tcmb",
    "Neff",
    "YHe",
    "tau_reio",
    "Omegak",
    "w_DE_0",
    "w_DE_a",
    "cs2_DE",
    "Nmnu",
    "mnu",
)
"""The cosmology-dictionary keys :func:`solve_eb` differentiates with respect to.

Every key the solve reads. They reach it through the row's directed inputs
(:data:`DIRECTED_ROW_NAMES`): the background densities, the recombination
block's thermal inputs, the reionization redshift, the curvature, the
dark-energy fluid and the massive neutrinos. The forward sensitivities carry
them through both the hierarchy and the recombination history. The
derivative is complete: the conformal age ``tau0`` at which a trajectory
stops, the reionization redshift found for ``tau_reio``, the initial state
and, at scale factors, the conformal times of them move with the
parameters. ``w_DE_0``, ``w_DE_a`` and ``cs2_DE`` need a layout with the
dark-energy fluid, ``Nmnu`` and ``mnu`` one with massive neutrinos. The other
keys (``A_s``, ``n_s``, ``k_p``) do not enter the solve."""

PARAMETER_DEFAULTS = {
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omegak": 0.0,
    "Nmnu": 0.0,
    "mnu": 0.0,
    "tau_reio": 0.0,
    "k_p": 0.05,
}
"""The values of the keys a parameter dictionary may leave out (:func:`_as_cosmology`)."""


def _cosmology_dictionary(cosmology) -> dict:
    """Return a cosmology's parameter dictionary, given a dictionary or a cosmology object."""

    if isinstance(cosmology, dict):
        return {**PARAMETER_DEFAULTS, **cosmology}
    d = cosmology.to_background_params()
    d["k_p"] = cosmology.k_pivot
    return d


def _split_cosmologies(cosmologies):
    """Return ``(fixed, theta)``: each cosmology's plain and differentiable parameters.

    ``fixed`` holds one dictionary per cosmology of the keys outside
    :data:`DIFFERENTIABLE_PARAMETERS`, as given: numbers, or JAX values under
    ``jit`` (:func:`solve_eb` refuses a derivative along them). ``theta`` is
    the ``(n_cosmologies, 5)`` array of the differentiable keys, traced or not.
    """

    fixed, rows = [], []
    for cosmology in cosmologies:
        d = _cosmology_dictionary(cosmology)
        fixed.append({k: v for k, v in d.items() if k not in DIFFERENTIABLE_PARAMETERS})
        values = [d[k] for k in DIFFERENTIABLE_PARAMETERS]
        if any(isinstance(v, jax.Array) for v in values):
            rows.append(jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values]))
        else:
            rows.append(np.asarray(values, dtype=np.float64))
    if all(isinstance(r, np.ndarray) for r in rows):
        # plain numbers: one host array, no per-cosmology device work
        return tuple(fixed), np.stack(rows)
    return tuple(fixed), jnp.stack(rows)


LAYOUT_KEYS = ("w_DE_0", "w_DE_a", "Nmnu")
"""The parameter-dictionary keys that choose a cosmology's perturbation layout."""


def _batch_layout(dictionaries) -> PerturbationLayout:
    """Return the layout the cosmologies' :data:`LAYOUT_KEYS` imply, which must be numbers."""

    defaults = {"w_DE_0": -1.0, "w_DE_a": 0.0, "Nmnu": 0.0}
    layouts = set()
    for d in dictionaries:
        values = {key: d.get(key, defaults[key]) for key in LAYOUT_KEYS}
        if any(isinstance(v, jax.core.Tracer) for v in values.values()):
            raise ValueError(
                f"{LAYOUT_KEYS} choose the perturbation layout and are traced: "
                "pass layout= to solve_eb"
            )
        layouts.add(
            _layout_for(
                SimpleNamespace(
                    w_DE_0=float(values["w_DE_0"]),
                    w_DE_a=float(values["w_DE_a"]),
                    num_massive_neutrinos=float(values["Nmnu"]),
                )
            )
        )
    if len(layouts) > 1:
        raise ValueError(
            "all cosmologies in a batch must share a perturbation layout "
            "(same dark-energy / massive-neutrino settings)"
        )
    return layouts.pop()


def _input_vector(fixed: dict, theta_row, xp=jnp) -> jax.Array:
    """Return the inputs of :data:`DIRECTED_NAMES` from a cosmology's parameters, in JAX (or ``xp``)."""

    param = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta_row))}
    d = density_parameters(param, xp=xp)
    return xp.stack(
        [
            d["grhog"],
            d["grhornomass"],
            d["grhoc"],
            d["grhob"],
            d["grhov"],
            param["YHe"],
            param["Tcmb"],
            param["H0"],
            param["Omegab"],
        ]
    )


def _named_error_weights(weights, layout, k_launch):
    """Turn ``{state name: weight}`` into solver error weights ``(n_traj, nvar)`` in launch order.

    Each weight is a number or a function of the trajectories' wave numbers
    (``k_launch``, launch order) that returns one per trajectory. Unnamed states
    keep weight one. The names are the layout's
    (:func:`~discoeb.eb_model.layout_state_names`), so a caller never needs the
    state's index order or the launch's trajectory order.
    """

    names = layout_state_names(layout)
    unknown = [n for n in weights if n not in names]
    if unknown:
        raise KeyError(f"{unknown} are not states of this layout: {names[:6]} ...")
    k_launch = np.asarray(k_launch, dtype=np.float64)
    out = np.ones((len(k_launch), len(names)))
    for name, weight in weights.items():
        out[:, names.index(name)] = weight(k_launch) if callable(weight) else weight
    return out


def _kept_values(
    prepared,
    k_values,
    tau_grid,
    names,
    extra=(),
    *,
    drop_first: bool = True,
    error_weights=None,
    tau_end_of=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
):
    """Return the hooked launches of :func:`solve_eb` as a function of the directed inputs.

    The result, ``of_inputs(x)``, takes ``x``, the ``(n_cosmologies, 9)``
    inputs of :data:`DIRECTED_NAMES`, and returns the kept quantities as
    ``(n_cosmologies, len(names), n_k, n_tau)`` in the caller's wave-mode order.
    It is a ``jax.custom_jvp`` in ``x``. The plain hooked launch gives the
    value. A forward-sensitivity launch gives the tangent: it replicates every
    trajectory once per input, and each replica carries the single column
    ``S = dy / d eps`` of its unit direction through ``IX_EPS`` and ``IX_DIR``.
    Its hook is the tangent hook of :func:`build_keep_hook`, which stores each
    kept quantity and its derivative along the column. So the kernel forms the
    derivative too and stores no history.

    The joint launch uses the fp64 LU and keeps the sensitivities out of step
    control. Its primal agrees with the plain fp32-LU one to ~1e-6 of a
    quantity's scale. The reionization redshift stays that of ``prepared``. So
    does ``tau0``, unless ``tau_end_of`` maps ``x`` to each cosmology's
    conformal age. The trajectories then stop at the age of the given inputs, in
    value only: the derivative is still the one at fixed ``tau0``.
    """

    layout = prepared.layout
    n_cosmo = len(prepared.cosmologies)
    names = tuple(names)
    extra = tuple(extra)
    settings: dict[str, Any] = dict(
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )
    nvar = prepared.model.nvar

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_grid)
    if isinstance(error_weights, dict):
        error_weights = _named_error_weights(
            error_weights, layout, np.repeat(k_sorted, n_cosmo)
        )
    n_k, n_grid, n_keep = len(k_sorted), len(tau_grid), len(names)
    first = 1 if drop_first else 0
    columns = jnp.asarray(DIRECTED_COLUMNS)
    n_dir = len(DIRECTED_COLUMNS)
    t_span_j = jnp.asarray(t_span)

    def with_inputs(base, x, replicas=1):
        # Row k_idx * n_cosmo + ci (times replicas) takes cosmology ci's row of x.
        def per_row(v):
            return jnp.repeat(jnp.tile(v, (n_k, 1)), replicas, axis=0)

        base = jnp.asarray(base).at[:, columns].set(per_row(x))
        if tau_end_of is not None:
            tau_end = jnp.asarray(tau_end_of(jax.lax.stop_gradient(x)))
            base = base.at[:, IX_TAU_END].set(per_row(tau_end[:, None])[:, 0])
        return base

    def arrange(rows):
        # Rows are k-major (row = k_idx * n_cosmo + cosmology_idx) and each
        # hook row is save-major, so this reshape is a view. The transpose to
        # cosmology-major copies only the kept values.
        values = rows.reshape(n_k, n_cosmo, n_grid, n_keep)
        values = jnp.transpose(values[:, :, first:, :], (1, 3, 0, 2))
        if np.any(order != np.arange(n_k)):
            values = values[:, :, jnp.asarray(np.argsort(order)), :]
        return values

    @jax.custom_jvp
    def of_inputs(x):
        _, out = _launch(
            prepared,
            with_inputs(params, x),
            jnp.asarray(y0),
            t_span_j,
            error_weights=error_weights,
            save_hook=_keep_hook(layout, extra, names, False),
            hook_size=n_keep * n_grid,
            save_history=False,
            **settings,
        )
        return arrange(out)

    @of_inputs.defjvp
    def of_inputs_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        # Replica j of trajectory i is row i * n_dir + j, along the unit
        # direction e_j. Each carries the single column S = dy / d eps.
        params_rep = (
            jnp.repeat(jnp.asarray(params), n_dir, axis=0)
            .at[:, IX_DIR : IX_DIR + n_dir]
            .set(np.tile(np.eye(n_dir), (n_k * n_cosmo, 1)))
        )
        _, out = _launch(
            prepared,
            with_inputs(params_rep, x, n_dir),
            jnp.repeat(jnp.asarray(y0), n_dir, axis=0),
            t_span_j,
            error_weights=None
            if error_weights is None
            else np.repeat(np.asarray(error_weights), n_dir, axis=0),
            save_hook=_keep_hook(layout, extra, names, True),
            hook_size=2 * n_keep * n_grid,
            save_history=False,
            lu_precision="fp64",
            # Thread-local memory, not registers, bounds the joint system's
            # occupancy. An uncapped register budget halves the resident
            # threads the driver reserves local memory for.
            max_registers=None,
            sensitivity=eps_sensitivity(nvar),
            **settings,
        )
        out = out.reshape(n_k * n_cosmo, n_dir, n_grid, 2, n_keep)
        # Every replica integrates the same state, so the first's primal serves.
        values = arrange(out[:, 0, :, 0, :])
        # sens[i, j] = d q_i / d x_j at every save. The contraction with the
        # input tangent is linear, so jax.grad transposes it.
        dx_rows = jnp.tile(dx, (n_k, 1))
        tangent = arrange(jnp.einsum("ijsq,ij->isq", out[:, :, :, 1, :], dx_rows))
        return values, tangent

    return of_inputs


def solve_eb(
    cosmologies,
    k_values,
    tau=None,
    *,
    a=None,
    keep=None,
    extra=(),
    inputs=None,
    layout=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    error_weights=None,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
) -> EBSolution:
    """Solve the Einstein-Boltzmann system of ``cosmologies`` and keep ``keep`` at ``tau``.

    Like :func:`solve_perturbation_history_batch`, this runs one modax Rodas5P
    launch over every (wave mode, cosmology) pair. Unlike it, the launch records
    only the quantities named in ``keep`` at the requested times, through
    modax's save hook. The kernel evaluates them from the dense-output state at
    each save and stores ``len(keep)`` numbers, so the device holds ``n_k *
    n_cosmologies * len(tau) * len(keep)`` values and never the history. Keeping
    only the comoving curvature perturbation at 1000 times for 128 modes and 128
    cosmologies stores 130 MB instead of the 7 GB of the 55-variable history.

    Parameters
    ----------
    cosmologies
        One cosmology (a parameter dictionary or a cosmology object) or a
        sequence of them, all with one perturbation layout. Any value may be a
        JAX array, traced or not, as under ``jit``. Only those of
        :data:`DIFFERENTIABLE_PARAMETERS` may be differentiated.
    k_values
        Wave numbers in 1/Mpc, in any order.
    tau
        Strictly increasing conformal times in Mpc at which to record, starting
        at or after ``TAU_START``. Every launch starts at ``TAU_START``, so a
        first time equal to it records the initial state. A cosmology's
        trajectory ends at its conformal age ``tau0``, and later times hold its
        state today. A last time past every ``tau0`` therefore returns the
        present-day values of the whole batch. Give ``tau`` or ``a``.
    a
        Strictly increasing scale factors, from :data:`A_MIN` to 1, at which to
        record instead. Each cosmology records at its own conformal times of
        them (``EBSolution.tau``, from the background quadrature, linear in
        ``ln a``), and ``a = 1`` is its ``tau0``. The derivative is then the
        one at fixed ``a``.
    keep
        Names of the quantities to record: state elements
        (:func:`~discoeb.eb_model.layout_state_names`, or the recombination
        block's ``lna``, ``thermo_0``, ``thermo_1``, ``thermo_2``, ``kappa``) or
        quantities the joint model provides, such as the comoving curvature
        perturbation ``R``, the shear ``sigma``, the expansion rate ``adotoa`` or
        the opacity. Default: every element of the perturbation state, which
        amounts to the hierarchy's full history.
    extra
        Components to append to the model for the hook: stateless
        :class:`~discoeb.component.Component` subclasses whose providers read the
        model's quantities and add their own (the line-of-sight sources of
        :mod:`discocmb.cmb`). The hook cache tells them apart by class.
    inputs
        The ``(n_cosmologies, 9)`` directed inputs of :data:`DIRECTED_NAMES`
        (the five background densities and the recombination block's ``YHe``,
        ``Tcmb``, ``H0``, ``Omegab``), replacing those the cosmologies imply
        (:func:`directed_inputs`). They are differentiable in their own right:
        the solution becomes a function of them, at fixed background endpoints
        of the cosmologies.
    layout
        The batch's :class:`PerturbationLayout`. Needed when ``w_DE_0``,
        ``w_DE_a`` or ``Nmnu``, which choose it, are traced. Otherwise it is
        the one they imply.
    rtol, atol, first_step, max_steps, error_weights, thermo_weight
        Solver settings, as in :func:`solve_perturbation_history_batch`.
        ``error_weights`` is either ``{state name: weight}`` or modax's array
        ``(n_k * n_cosmologies, nvar)`` in launch order (wave modes ascending,
        cosmologies within). In the dictionary, each weight is a number or a
        function of the trajectories' wave numbers that returns one per
        trajectory, and unnamed states keep weight one.

    Notes
    -----
    The hook compiles once per batch shape and set of names, like the
    right-hand side. The returned ``values`` are differentiable in the
    cosmologies' :data:`DIFFERENTIABLE_PARAMETERS` under ``jax.jvp``,
    ``jax.jacfwd`` and ``jax.grad``, and the solve runs under ``jit``: with
    concrete values the host prepares the batch (cached), with traced ones JAX
    does, the endpoints, rows and initial states included. The derivative runs
    through the row's directed inputs (:data:`DIRECTED_ROW_NAMES`). A
    forward-sensitivity launch carries it, with a hook that stores each kept
    quantity's derivative beside it, so it also costs one joint launch and no
    history. It is the complete derivative, at fixed
    ``tau`` or fixed ``a``: one launch carries a replica per parameter with a
    tangent, which moves the directed inputs, ``z_re`` and the initial state
    along it, and one replica whose tangent is each quantity's rate, for the
    save times that move with the parameters (``tau0``, the conformal times
    of ``a``). With ``inputs`` it is :func:`_kept_values`' derivative instead,
    at fixed background endpoints.
    """

    dicts = [_cosmology_dictionary(c) for c in _as_batch(cosmologies)]
    n_cosmo = len(dicts)
    fixed = [
        {k: v for k, v in d.items() if k not in DIFFERENTIABLE_PARAMETERS}
        for d in dicts
    ]
    # One leaf per key, so a key without a tangent is a symbolic zero to the
    # rule below and costs no replica.
    theta = {
        key: _stacked([d[key] for d in dicts]) for key in DIFFERENTIABLE_PARAMETERS
    }

    if (tau is None) == (a is None):
        raise ValueError("give the save times as tau or as a, not both")
    if a is not None:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.size == 0 or not np.all(np.isfinite(a)):
            raise ValueError("a must be a non-empty array of finite scale factors")
        if a[0] < A_MIN or a[-1] > 1.0 or np.any(np.diff(a) <= 0.0):
            raise ValueError(
                f"a must be strictly increasing, from A_MIN = {A_MIN} to 1"
            )
        if inputs is not None:
            raise ValueError("inputs= takes its save times as tau")
        # The launch records TAU_START first, before every a.
        drop_first, tau_grid = True, None
        n_grid = len(a) + 1
    else:
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        if tau.size == 0 or not np.all(np.isfinite(tau)):
            raise ValueError("tau must be a non-empty array of finite conformal times")
        if tau[0] < TAU_START or np.any(np.diff(tau) <= 0.0):
            raise ValueError(
                f"tau must be strictly increasing and from TAU_START = {TAU_START} on"
            )
        # The launch starts at TAU_START, so the hook always records it. Drop
        # that row unless the caller asked for it.
        drop_first = bool(tau[0] > TAU_START)
        tau_grid = np.concatenate(([TAU_START], tau)) if drop_first else tau
        n_grid = len(tau_grid)
    settings: dict[str, Any] = dict(
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )

    def dictionaries(theta, fixed):
        return [
            {**fixed[i], **{key: theta[key][i] for key in DIFFERENTIABLE_PARAMETERS}}
            for i in range(n_cosmo)
        ]

    if layout is None:
        layout = _batch_layout(dictionaries(theta, fixed))
    # keys that act on what the layout leaves out, so refuse a derivative
    left_out_keys = (
        () if layout.enable_dark_energy else ("w_DE_0", "w_DE_a", "cs2_DE")
    ) + (() if layout.has_massive_neutrinos else ("Nmnu", "mnu"))
    names = tuple(layout_state_names(layout) if keep is None else keep)
    if len(set(names)) != len(names):
        raise ValueError(f"keep names a quantity twice: {names}")
    k_values = np.asarray(k_values, dtype=np.float64).reshape(-1)
    n_k, n_keep = len(k_values), len(names)
    order = np.argsort(k_values, kind="stable")
    if isinstance(error_weights, dict):
        error_weights = _named_error_weights(
            error_weights, layout, np.repeat(k_values[order], n_cosmo)
        )

    def prepared_of(theta, fixed):
        """The prepared solve: on the host for numbers (cached), in JAX for traced values."""

        if any(
            isinstance(v, jax.core.Tracer)
            for v in jax.tree_util.tree_leaves((theta, fixed))
        ):
            cosmos = tuple(
                _as_cosmology(d, lambda v: jnp.asarray(v, dtype=jnp.float64))
                for d in dictionaries(theta, fixed)
            )
            return _prepare_traced(cosmos, layout)
        numbers = {k: np.asarray(v, dtype=np.float64) for k, v in theta.items()}
        prepared = _prepare_solve(
            tuple(_as_cosmology(d) for d in dictionaries(numbers, fixed))
        )
        if prepared.layout != layout:
            raise ValueError(
                f"layout= {layout} is not the cosmologies' own {prepared.layout}"
            )
        return prepared

    def save_times(prepared):
        """Each cosmology's save times, ``(n_cosmologies, n_grid)``."""

        if a is None:
            return np.broadcast_to(tau_grid, (n_cosmo, n_grid))
        tau_a = _conformal_times(
            _stack_background_params(prepared.cosmologies), jnp.asarray(a)
        )
        times = jnp.concatenate([jnp.full((n_cosmo, 1), TAU_START), tau_a], axis=1)
        return times if isinstance(prepared.tau0, jax.Array) else np.asarray(times)

    def inputs_of(theta, fixed):
        # JAX arithmetic throughout, also for parameters given as plain numbers
        theta = jnp.stack(
            [
                jnp.asarray(theta[key], dtype=jnp.float64)
                for key in DIFFERENTIABLE_PARAMETERS
            ],
            axis=1,
        )
        keys = set(fixed[0])
        if (
            isinstance(theta, jax.core.Tracer)
            or n_cosmo == 1
            or any(set(f) != keys for f in fixed)
        ):
            return jnp.stack(
                [_input_vector(fixed[i], theta[i]) for i in range(n_cosmo)]
            )
        # Concrete parameters: one set of eager elementwise JAX operations for
        # the whole batch matches each cosmology alone bit for bit. A last-bit
        # difference in a density can move a step acceptance.
        columns = {
            k: jnp.asarray([f[k] for f in fixed], dtype=jnp.float64) for k in keys
        }
        return _input_vector(columns, [theta[:, j] for j in range(theta.shape[1])]).T

    def launch_inputs(theta, fixed):
        """The launch's rows, initial states and save times, ``row = k_idx * n_cosmo + c``.

        A JAX function of the parameters when they are traced: the rows'
        inputs, ``tau0`` and ``z_re``, the initial states and the conformal
        times of ``a`` all move with them.
        """

        prepared = prepared_of(theta, fixed)
        params, y0, _, _, _ = _pack_batch(prepared, k_values)
        x = jnp.tile(inputs_of(theta, fixed), (n_k, 1))
        params = jnp.asarray(params).at[:, DIRECTED_COLUMNS].set(x)
        t_save = jnp.tile(jnp.asarray(save_times(prepared)), (n_k, 1))
        return (
            prepared,
            params,
            jnp.asarray(y0),
            jnp.minimum(t_save, params[:, IX_TAU_END, None]),
        )

    def arrange(rows):
        # Rows are k-major (row = k_idx * n_cosmo + cosmology_idx) and each
        # hook row is save-major, so this reshape is a view. The transpose to
        # cosmology-major copies only the kept values.
        values = rows.reshape(n_k, n_cosmo, n_grid, n_keep)
        values = jnp.transpose(values[:, :, int(drop_first) :, :], (1, 3, 0, 2))
        if np.any(order != np.arange(n_k)):
            values = values[:, :, jnp.asarray(np.argsort(order)), :]
        return values

    def prepare(theta, fixed):
        """The inputs-level kept-value function of :func:`_kept_values`, for ``inputs``."""

        return _kept_values(
            prepared_of(theta, fixed),
            k_values,
            tau_grid,
            names,
            extra,
            drop_first=drop_first,
            error_weights=error_weights,
            **settings,
        )

    def solution(values):
        # tau0 and the conformal times of a are JAX functions of traced
        # parameters, so they carry their derivatives too
        prepared = prepared_of(theta, fixed)
        tau0 = prepared.tau0
        return EBSolution(
            names=names,
            k=k_values,
            tau=tau if tau is not None else save_times(prepared)[:, 1:],
            tau0=tau0 if isinstance(tau0, jax.Array) else np.asarray(tau0),
            values=values,
            layout=layout,
            a=a,
        )

    if inputs is not None:
        # The given inputs are the solve's variables. The cosmologies set
        # everything else, the background endpoints included, as numbers.
        inputs = jnp.asarray(inputs, dtype=jnp.float64)
        if inputs.shape != (n_cosmo, len(DIRECTED_NAMES)):
            raise ValueError(
                f"inputs must be (n_cosmologies, {len(DIRECTED_NAMES)}), got {inputs.shape}"
            )
        return solution(prepare(jax.lax.stop_gradient(theta), fixed)(inputs))

    @jax.custom_jvp
    def kept_of(theta, fixed):
        prepared, params, y0, t_rows = launch_inputs(theta, fixed)
        _, out = _launch(
            prepared,
            params,
            y0,
            jax.lax.stop_gradient(t_rows),
            error_weights=error_weights,
            save_hook=_keep_hook(layout, tuple(extra), names, False),
            hook_size=n_keep * n_grid,
            save_history=False,
            **settings,
        )
        return arrange(out)

    def kept_of_jvp(primals, tangents):
        """One derivative launch: a replica per key with a tangent, and one for the time.

        Replica ``j`` of a trajectory moves along key ``j`` everything the
        launch reads: its row's directed inputs and ``z_re`` (the ``dir``
        block), and its initial state (``dy0``, which the shifted blocks of
        :func:`joint_kernel` add to the state). Its tangent hook stores
        ``dq/d theta_j`` at the save's fixed time. The last replica moves no
        input but the time, so it stores each quantity's rate ``dq/dtau``.
        The save time itself, ``min(t_save, tau0)``, moves with the
        parameters too (the conformal time of an ``a``, the end of a
        trajectory), which adds the rate times its derivative.
        """

        (theta, fixed), (dtheta, dfixed) = primals, tangents
        for f, df in zip(fixed, dfixed):
            for key in f:
                if not isinstance(df[key], SymbolicZero):
                    raise ValueError(
                        f"{key!r} cannot be differentiated through solve_eb: the "
                        f"differentiable parameters are {DIFFERENTIABLE_PARAMETERS}"
                    )
        keys = [
            k
            for k in DIFFERENTIABLE_PARAMETERS
            if not isinstance(dtheta[k], SymbolicZero)
        ]
        left_out = [k for k in keys if k in left_out_keys]
        if left_out:
            raise ValueError(
                f"{left_out[0]!r} cannot be differentiated through solve_eb with "
                f"{layout}: it acts on what the layout leaves out"
            )
        theta = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in theta.items()}
        prepared = prepared_of(theta, fixed)
        (params, y0, t_rows), linear = jax.linearize(
            lambda t: launch_inputs(t, fixed)[1:], theta
        )
        n_dir, n_rep = len(keys), len(keys) + 1
        units = {
            k: jnp.zeros((n_dir, n_cosmo))
            if k not in keys
            else jnp.zeros((n_dir, n_cosmo)).at[keys.index(k)].set(1.0)
            for k in DIFFERENTIABLE_PARAMETERS
        }
        d_params, d_y0, d_t = jax.vmap(linear)(units)

        n_traj, nvar = y0.shape
        n_row = len(DIRECTED_ROW_COLUMNS)
        directions = jnp.concatenate(
            [
                d_params[:, :, jnp.asarray(DIRECTED_ROW_COLUMNS)],
                jnp.zeros((1, n_traj, n_row)),
            ]
        )
        dy0 = jnp.concatenate([d_y0, jnp.zeros((1, n_traj, nvar))])
        time = jnp.zeros((n_rep, n_traj, 1)).at[n_dir].set(1.0)
        rows = (
            jnp.broadcast_to(params, (n_rep, n_traj, N_PARAM))
            .at[:, :, IX_DIR : IX_DIR + n_row]
            .set(directions)
        )
        # Replica r of trajectory i is row i * n_rep + r.
        rows = jnp.swapaxes(jnp.concatenate([rows, dy0, time], axis=2), 0, 1)
        rows = rows.reshape(n_traj * n_rep, -1)
        _, out = _launch(
            prepared._replace(ode_fn=joint_kernel(layout, shift=True)[1]),
            rows,
            jnp.repeat(y0, n_rep, axis=0),
            jax.lax.stop_gradient(jnp.repeat(t_rows, n_rep, axis=0)),
            error_weights=None
            if error_weights is None
            else np.repeat(np.asarray(error_weights), n_rep, axis=0),
            save_hook=_keep_hook(layout, tuple(extra), names, True, True),
            hook_size=2 * n_keep * n_grid,
            save_history=False,
            lu_precision="fp64",
            # Thread-local memory, not registers, bounds the joint system's
            # occupancy. An uncapped register budget halves the resident
            # threads the driver reserves local memory for.
            max_registers=None,
            sensitivity=eps_sensitivity(nvar, rows.shape[1]),
            **settings,
        )
        out = out.reshape(n_traj, n_rep, n_grid, 2, n_keep)
        # Every replica integrates the same state, so the first's primal serves.
        values = arrange(out[:, 0, :, 0, :])
        rate = out[:, n_dir, :, 1, :]
        along = (
            out[:, :n_dir, :, 1, :] + rate[:, None] * jnp.moveaxis(d_t, 0, 1)[..., None]
        )
        # The contraction with the parameters' tangent is linear, so jax.grad
        # transposes it.
        dtheta_rows = jnp.tile(jnp.stack([dtheta[k] for k in keys], axis=1), (n_k, 1))
        tangent = arrange(jnp.einsum("ijsq,ij->isq", along, dtheta_rows))
        return values, tangent

    kept_of.defjvp(kept_of_jvp, symbolic_zeros=True)
    return solution(kept_of(theta, fixed))


def _stacked(values):
    """Stack one value per cosmology: a numpy array of numbers, a JAX array if any is one."""

    if any(isinstance(v, jax.Array) for v in values):
        return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values])
    return np.asarray(values, dtype=np.float64)


def directed_inputs(cosmologies) -> np.ndarray:
    """Return the ``(n_cosmologies, 9)`` directed inputs of :data:`DIRECTED_NAMES` the cosmologies imply.

    These are the five background densities and the recombination block's
    ``YHe``, ``Tcmb``, ``H0`` and ``Omegab``: the variables :func:`solve_eb`
    takes as ``inputs`` and differentiates with respect to.
    """

    fixed, theta = _split_cosmologies(_as_batch(cosmologies))
    return np.stack(
        [
            np.asarray(_input_vector(fixed[i], jnp.asarray(theta[i])))
            for i in range(len(fixed))
        ]
    )


def conformal_ages(cosmologies) -> np.ndarray:
    """Return each cosmology's conformal age ``tau0`` (Mpc), where its trajectories stop.

    Reads the same cached background endpoints as :func:`solve_eb`, so a solve
    of the same cosmologies right after reuses them.
    """

    return np.asarray(_prepare_solve(_as_batch(cosmologies)).tau0, dtype=np.float64)


def state_names(cosmology) -> tuple[str, ...]:
    """Return the perturbation state names of a cosmology's layout, for ``keep`` and ``error_weights``."""

    return layout_state_names(_layout_for(_as_cosmology(cosmology)))


as_cosmology = _as_cosmology
"""Normalise a cosmology given as a parameter dictionary (``Omegam``, ``Omegab``,
``H0``, ``Tcmb``, ``YHe``, ``Neff``, ``A_s``, ``n_s``, ... ) to the solver's
cosmology object. Returns one already normalised as it is."""
