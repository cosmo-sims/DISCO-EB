"""The perturbation model a layout enables, its state names and its sparsity."""

from __future__ import annotations

import functools

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
