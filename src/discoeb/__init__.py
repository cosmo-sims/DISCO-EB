"""DISCO-EB and the interface the CMB code (DISCO-CMB) may use.

DISCO-EB solves the background, the thermal history (RECFAST with
reionization) and the linear perturbation hierarchy of each wave mode on the
GPU. :mod:`discoeb.components.component` holds the component framework,
:mod:`discoeb.components.species`, :mod:`discoeb.components.interactions` and
:mod:`discoeb.components.gravity` the equations, :mod:`discoeb.models` the cosmological
models built from them (``LCDM``, ``W0WACDM``, ...),
:mod:`discoeb.components.recombination` the thermal history, :mod:`discoeb.solver.background`
the expansion history and :mod:`discoeb.solver.kernel` the batching and modax
launches behind :func:`solve_eb` and the matter power spectrum. The CMB code lives in the separate package ``discocmb`` (DISCO-CMB) and turns
this output into angular power spectra: line-of-sight sources, grids,
transfer functions, projection and lensing. It imports only from this
package itself. DISCO-EB does not depend on it.

**The contract.** A caller asks for *named quantities at chosen conformal
times* of a batch of cosmologies and wave modes. The kernel launch evaluates
them from the solver's dense output and returns those values, never the
state history:

* :func:`solve_eb` ``(cosmologies, k, tau, keep=names, extra=components)``,
  with cosmologies of one model such as ``[LCDM(H0=..., ...), ...]``,
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
* Differentiation: give the cosmologies' :data:`DIFFERENTIABLE_PARAMETERS`
  as JAX values (those of :func:`differentiable_parameters` for the model),
  or pass the
  :data:`DIRECTED_NAMES` inputs (:func:`directed_inputs`, the model's, the
  superset or the first nine) as ``inputs``, and ``values`` becomes
  differentiable (``jax.jvp``, ``jax.grad``) through forward sensitivities
  the kernel computes. The derivatives hold ``tau0`` fixed.

:func:`background_history`, :func:`density_coefficients` and
:func:`conformal_ages` give the background for building grids.

**Models and cosmologies.** A model is a class whose bases are its
components (:class:`Model`, :mod:`discoeb.models`), and a cosmology an
instance of it: ``LCDM(H0=67.32, Omegam=0.3158, ...)``. The components
declare the parameters, so the constructor takes exactly the model's and a
type checker sees them. ``Model.spec`` is the :class:`ModelSpec` the solver
caches its kernel on. :func:`as_cosmology` gives a cosmology's parameters in
a hashable form.
"""

# First: double precision and JAX's GPU memory policy, before anything builds
# an array or touches the device.
from . import _jax_config  # noqa: F401
from .components.component import Component
from .components.recombination import THERMO_ERROR_WEIGHT
from .models import (
    LCDM,
    LCDM_MNU,
    W0WACDM,
    W0WACDM_MNU,
    Model,
    ModelSpec,
)
from .models.cosmology import DIFFERENTIABLE_PARAMETERS, as_cosmology
from .observables.power_spectrum import (
    matter_power_spectrum,
)
from .solver.background import TAU_START, background_history, density_coefficients
from .solver.kernel import (
    DIRECTED_NAMES,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    joint_model,
)
from .solver.solver import (
    EBSolution,
    conformal_ages,
    differentiable_parameters,
    directed_inputs,
    solve_eb,
    state_names,
)

__all__ = [
    "Component",
    "Model",
    "ModelSpec",
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
