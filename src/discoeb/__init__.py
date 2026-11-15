"""DISCO-EB and the interface the CMB code (DISCO-CMB) may use.

DISCO-EB solves the background, the thermal history (RECFAST with
reionization) and the linear perturbation hierarchy of each wave mode on the
GPU. :mod:`discoeb.component` holds the component framework,
:mod:`discoeb.species`, :mod:`discoeb.interactions` and
:mod:`discoeb.gravity` the equations, :mod:`discoeb.models` the cosmological
models built from them (``LCDM``, ``W0WACDM``, ...),
:mod:`discoeb.recombination` the thermal history, :mod:`discoeb.background`
the expansion history and :mod:`discoeb.kernel` the batching and modax
launches behind :func:`solve_eb` and the matter power spectrum. The CMB code lives in the separate package ``discocmb`` (DISCO-CMB) and turns
this output into angular power spectra: line-of-sight sources, grids,
transfer functions, projection and lensing. It imports only from this
package itself. DISCO-EB does not depend on it.

**The contract.** A caller asks for *named quantities at chosen conformal
times* of a batch of cosmologies and wave modes. The kernel launch evaluates
them from the solver's dense output and returns those values, never the
state history:

* :func:`solve_eb` ``(LCDM(cosmologies), k, tau, keep=names,
  extra=components)``
  returns an :class:`EBSolution` whose ``values`` are ``(n_cosmologies,
  len(keep), n_k, n_tau)``, with each cosmology's conformal age ``tau0``.
  Past its own ``tau0``, a cosmology's trajectories hold their state today.
* A name is a state element (:func:`state_names`, including the thermal
  block's ``lna``, ``thermo_0..2`` and the optical depth ``kappa``) or any
  quantity the model provides (``adotoa``, ``opacity``, ``sigma``,
  ``dgrho``, ``clxg``, ``polter``, ``R`` ...). ``extra`` extends the
  vocabulary: stateless :class:`Component` subclasses whose providers read
  the model's quantities by name and add their own, compiled into the same
  kernel. The CMB's line-of-sight sources work this way, so both parts share
  one component model and nothing of the CMB enters the hierarchy's
  right-hand side.
* Error weights go by state name (``{"theta_2": f(k)}``), so the caller
  needs neither the state's index order nor the launch order.
* Differentiation: pass :data:`DIFFERENTIABLE_PARAMETERS` as JAX values
  (those of :func:`differentiable_parameters` for the model), or the
  :data:`DIRECTED_NAMES` inputs (:func:`directed_inputs`, the model's, the
  superset or the first nine) as ``inputs``, and ``values`` becomes
  differentiable (``jax.jvp``, ``jax.grad``) through forward sensitivities
  the kernel computes. The derivatives hold ``tau0`` fixed.

:func:`background_history`, :func:`density_coefficients` and
:func:`conformal_ages` give the background for building grids. A cosmology
is a parameter dictionary or an object :func:`as_cosmology` returns.
"""

# First: sets JAX's GPU memory policy before anything touches the device.
from . import _jax_memory  # noqa: F401
from .background import TAU_START, background_history, density_coefficients
from .component import Component
from .cosmology import DIFFERENTIABLE_PARAMETERS, as_cosmology
from .kernel import (
    DIRECTED_NAMES,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    joint_model,
)
from .models import (
    LCDM,
    LCDM_MNU,
    W0WACDM,
    W0WACDM_MNU,
    CosmologyBatch,
    CosmologyModel,
)
from .power_spectrum import (
    matter_power_spectrum,
)
from .recombination import THERMO_ERROR_WEIGHT
from .solver import (
    EBSolution,
    conformal_ages,
    differentiable_parameters,
    directed_inputs,
    solve_eb,
    state_names,
)

__all__ = [
    "Component",
    "CosmologyBatch",
    "CosmologyModel",
    "DIFFERENTIABLE_PARAMETERS",
    "DIRECTED_NAMES",
    "EBSolution",
    "PERTURB_ATOL",
    "PERTURB_FIRST_STEP",
    "PERTURB_MAX_STEPS",
    "PERTURB_RTOL",
    "LCDM",
    "LCDM_MNU",
    "W0WACDM",
    "W0WACDM_MNU",
    "TAU_START",
    "THERMO_ERROR_WEIGHT",
    "as_cosmology",
    "background_history",
    "conformal_ages",
    "density_coefficients",
    "differentiable_parameters",
    "directed_inputs",
    "joint_model",
    "matter_power_spectrum",
    "solve_eb",
    "state_names",
]
