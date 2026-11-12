"""The perturbation model a layout enables, its state names and its sparsity."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from .component import Component, Model
from .gravity import SUMS, AdiabaticMode, Background, Curvature, Metric
from .interactions import ThomsonScattering
from .species import (
    Baryons,
    ColdDarkMatter,
    CosmologicalConstant,
    DarkEnergy,
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
    RadiationStreaming,
)


def perturbation_model(
    layout, thermal_history: Component | None = None, extra=()
) -> Model:
    """The components a :class:`~discoeb.perturbations.PerturbationLayout` enables.

    ``thermal_history`` provides ``a``, ``opacity``, ``cs2``, ``rhonu`` and
    ``pnu`` as functions of ``tau``. In the solve it is the recombination
    block of :func:`discoeb.perturbations.joint_kernel`, in a test anything.
    Without it the model still gives the state layout, the initial conditions
    and the sparsity pattern. The model appends the ``extra`` components (the
    line-of-sight sources of :mod:`discocmb.cmb`). The components take the
    layout's truncations as settings, so a layout built from the components'
    own (environment-loaded) settings comes back unchanged. The state order
    matches the layout's indices: metric, CDM, baryons, photon temperature
    and polarization, massless neutrinos, then the dark-energy fluid and the
    massive-neutrino bins when enabled.
    """

    components = [
        Background(),
        RadiationStreaming(),
        AdiabaticMode(),
        Curvature(),
        Metric(),
        ColdDarkMatter(),
        Baryons(),
        Photons(LMAX_G=layout.lmaxg, LMAX_POL=layout.lmaxpol),
        MasslessNeutrinos(LMAX_NR=layout.lmaxr),
    ]
    if thermal_history is not None:
        components.insert(0, thermal_history)
    if layout.enable_dark_energy:
        components.append(DarkEnergy())
    else:
        components.append(CosmologicalConstant())
    if layout.has_massive_neutrinos:
        components.append(MassiveNeutrinos(LMAX_NU=layout.lmaxnu, NQMAX=layout.nqmax))
    components.append(ThomsonScattering())
    components.extend(extra)
    return Model(components, sums=SUMS)


def perturbation_sparsity(layout) -> np.ndarray:
    """The ``(nnz, 2)`` index array of the Jacobian's nonzeros for ``layout``.

    modax's Rodas5P kernel uses the pattern twice. It colours the column
    intersection graph, so a Jacobian costs ``n_colours + 1`` Enzyme sweeps
    rather than ``nvar + 1``. And ``modax._sparse_direct`` orders it,
    factorises it symbolically and compiles a sparse LU and a pair of sparse
    triangular solves for that exact structure. :meth:`Model.sparsity`
    traces the pattern through the equations, resolving the providers on
    values that carry the state elements they depend on. So the tridiagonal
    free-streaming tails of :func:`stream` and the Einstein constraints that
    tie the metric, the fluids and the lowest multipoles together come
    straight from the code that writes them. The pattern is exact: the
    superset a colouring needs (a subset would silently corrupt entries
    where two columns of a colour group overlap) and no more. At the default
    layout it has 165 entries, 11 colours and 198 slots of ``L + U`` after
    AMD's fill-in (``tests/test_eb_sparsity.py``).

    The kernel factorises ``I/(h gamma) - J``. modax keeps a slot for every
    diagonal in its packed layout and resolves ``slot(i, i)`` through that
    table even where ``J`` has no entry (an undamped free-streaming
    multipole has none). This needs modax at or after commit 9f54d12
    (2026-11-01). Older versions put ``1/(h gamma)`` on another column's
    entry of the same colour.
    """

    return perturbation_model(layout).sparsity()


@functools.lru_cache(maxsize=None)
def layout_state_names(layout) -> tuple[str, ...]:
    """The state names of :func:`perturbation_model` in the layout's index order."""

    return tuple(perturbation_model(layout).state)


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
