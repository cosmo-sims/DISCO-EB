# Models

A model is the list of components a solve evolves: the species (`discoeb.species`), the interactions between them (`discoeb.interactions`) and the metric and Friedmann sum (`discoeb.gravity`). Applied to cosmologies (parameter dictionaries) it gives the batch every solve takes, as `solve_eb(LCDM(params), k, tau)`. `discoeb.models` defines `LCDM` and `W0WACDM`, each with a massive-neutrino variant (`LCDM_MNU`, `W0WACDM_MNU`). A model rejects a cosmology it cannot represent, such as `w_DE_0 = -0.9` under `LCDM`.

Models derive from each other:

```python
from discoeb.models import LCDM
from discoeb.species import CosmologicalConstant, DarkEnergy, MassiveNeutrinos

W0WACDM = LCDM.replace(CosmologicalConstant, DarkEnergy, name="w0waCDM")
LCDM_MNU = LCDM.add(MassiveNeutrinos, name="LCDM+mnu")
COARSE = LCDM.with_settings(LMAX_G=10, LMAX_NR=10)  # shorter hierarchies
```

# Keeping quantities

`solve_eb` records only what you name in `keep`. A name can be a state element, such as `clxc`, `theta_2` or `psi_0_1`, which `state_names(model)` lists. It can also be any quantity the model provides, such as the comoving curvature perturbation `R`, the shear `sigma`, the expansion rate `adotoa` or the Thomson `opacity`. The recombination block adds `xe`, `Tmat`, `cs2` and the optical depth `kappa`.

The device holds `n_cosmologies * n_k * n_tau * len(keep)` numbers and never the full state history.

## Derivatives

Give `Omegam`, `Omegab`, `H0`, `Tcmb` or `Neff` as JAX values and the kept values become differentiable. `jax.jvp`, `jax.jacfwd` and `jax.grad` each cost one joint launch, which forms the derivatives inside the kernel by forward sensitivities:

```python
import jax.numpy as jnp

def R_today(omega_m):
    param = {**cosmology, "Omegam": omega_m}
    return solve_eb(LCDM(param), k, tau, keep=("R",))["R"][0, :, -1]

dR = jax.jacfwd(R_today)(jnp.asarray(cosmology["Omegam"]))  # (64,)
```

The derivative holds the conformal age and the reionization redshift fixed. The other parameters can't be differentiated, and trying raises an error.

## Under `jit`

`solve_eb` traces under `jax.jit` with every parameter traced, so one compiled function serves every cosmology with the same keys. The wave numbers must be plain numbers, since they set the launch's shape:

```python
import jax

@jax.jit
def power_spectrum(cosmology):
    batch = LCDM([cosmology])
    solution = solve_eb(
        batch,
        k,
        conformal_ages(batch).max()[None],
        keep=MATTER_POWER_QUANTITIES,
        thermo_weight=MATTER_THERMO_WEIGHT,
    )
    return compute_matter_power_spectrum(solution, batch)[0]
```

Under `jit` a model can't check its cosmologies (for instance `w_DE_0 = -1` under `W0WACDM`), since their values aren't known yet.

## Extending the model

A new species or interaction is a `Component` subclass; add it to a model with `add` or `replace`. To keep quantities a model does not provide, `solve_eb(..., extra=...)` appends your own components. A stateless `Component` subclass declares providers whose parameter names are the quantities they read. DISCO-CMB defines its line-of-sight sources this way, so they compile into the same kernel.
