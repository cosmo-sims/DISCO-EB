"""The Einstein-Boltzmann interface: what the CMB code -- DISCO-CMB -- may use.

DISCO-EB solves the background, the thermal history (RECFAST with
reionization) and the linear perturbation hierarchy of each wave mode on the
GPU: :mod:`discoeb.component` (the component model), :mod:`discoeb.eb_model`
(the species and their equations), :mod:`discoeb.recombination` and
:mod:`discoeb.thermodynamics_recfast` (the thermal history) and
:mod:`discoeb.perturbations` (layouts, batching, the modax launches). The CMB
code, the separate package ``discocmb`` (DISCO-CMB), turns its output into
angular power spectra -- line-of-sight sources, grids, transfer functions,
projection and lensing. It imports from this module alone (and the generic
numerics of :mod:`discoeb.spline_interpolation`); DISCO-EB does not depend
on it.

**The contract.** A caller asks for *named quantities at chosen conformal
times* of a batch of cosmologies and wave modes, and gets them back
evaluated inside the kernel launch from the solver's dense output -- never the
state history:

* :func:`solve_eb` ``(cosmologies, k, tau, keep=names, extra=components)``
  returns an :class:`EBSolution` whose ``values`` are ``(n_cosmologies,
  len(keep), n_k, n_tau)``, with each cosmology's conformal age ``tau0``;
  past its own ``tau0`` a cosmology's trajectories hold their state today.
* A name is a state element (:func:`state_names`; the thermal block's
  ``lna``, ``thermo_0..2`` and the optical depth ``kappa``) or any quantity
  the model provides (``adotoa``, ``opacity``, ``sigma``, ``dgrho``,
  ``clxg``, ``polter``, ``R`` ...). A caller extends the vocabulary with
  ``extra``: stateless :class:`Component` subclasses whose providers read the
  model's quantities by name and add their own, compiled into the same
  kernel -- this is how the CMB's line-of-sight sources are defined, so the
  equations of both parts live in one component model and nothing of the
  CMB enters the hierarchy's right-hand side.
* Error weights are given by state name (``{"theta_2": f(k)}``), so the
  caller needs neither the state's index order nor the launch order.
* Differentiation: :data:`DIFFERENTIABLE_PARAMETERS` given as JAX values, or
  the nine :data:`DIRECTED_NAMES` inputs (:func:`directed_inputs`) passed as
  ``inputs``, make ``values`` differentiable by forward sensitivities formed
  in the kernel (``jax.jvp``, ``jax.grad``), at fixed ``tau0`` and
  reionization redshift.

The background for building grids comes from :func:`background_history`,
:func:`density_coefficients` and :func:`conformal_ages`; cosmologies are
parameter dictionaries or the objects :func:`as_cosmology` returns.
"""

from .component import Component
from .perturbations import (
    DIFFERENTIABLE_PARAMETERS,
    DIRECTED_NAMES,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    TAU_START,
    EBSolution,
    PerturbationLayout,
    as_cosmology,
    background_history,
    conformal_ages,
    density_coefficients,
    directed_inputs,
    joint_model,
    solve_eb,
    state_names,
)
from .recombination import THERMO_ERROR_WEIGHT

__all__ = [
    "Component",
    "DIFFERENTIABLE_PARAMETERS",
    "DIRECTED_NAMES",
    "EBSolution",
    "PERTURB_ATOL",
    "PERTURB_FIRST_STEP",
    "PERTURB_MAX_STEPS",
    "PERTURB_RTOL",
    "PerturbationLayout",
    "TAU_START",
    "THERMO_ERROR_WEIGHT",
    "as_cosmology",
    "background_history",
    "conformal_ages",
    "density_coefficients",
    "directed_inputs",
    "joint_model",
    "solve_eb",
    "state_names",
]
