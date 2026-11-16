# Models

A model is a class whose bases are its components: the species it evolves (`discoeb.components.species`), the interactions between them (`discoeb.components.interactions`) and the metric and Friedmann sum (`discoeb.components.gravity`). An instance of a model is a cosmology. `discoeb.models` defines `LCDM` and `W0WACDM`, each with a massive-neutrino variant (`LCDM_MNU`, `W0WACDM_MNU`).

The components declare the parameters, so a model's constructor takes exactly its own, by keyword. A type checker reads them from the declarations and flags a missing or unknown one, as the constructor does at run time. `help(LCDM)` lists them. A cosmology is immutable: `planck.replace(H0=70.0)` gives a new one. Dictionaries go through `LCDM(**params)`, or `LCDM.from_dict(params)` for one written for another model, whose extra keys must keep their defaults.

A new model is a class statement. A model among the bases stands for its components, and the order of the bases is the state order:

```python
from discoeb.models import LCDM
from discoeb.components.species import MassiveNeutrinos

class LCDM_MNU(LCDM, MassiveNeutrinos):
    """LCDM with massive neutrinos."""

class Coarse(LCDM, LMAX_G=10, LMAX_NR=10):  # shorter hierarchies
    """LCDM with fewer photon and neutrino multipoles."""
```

`LCDM.swap(CosmologicalConstant, DarkEnergy, name="w0waCDM")`, `LCDM.add(...)`, `LCDM.remove(...)` and `LCDM.with_settings(...)` derive models at run time, but a type checker only sees the parameters of a model written as a class statement.

# Keeping quantities

`solve_eb` records only what you name in `keep`. A name can be a state element, such as `clxc`, `theta_2` or `psi_0_1`, which `state_names(model)` lists. It can also be any quantity the model provides, such as the comoving curvature perturbation `R`, the shear `sigma`, the expansion rate `adotoa` or the Thomson `opacity`. The recombination block adds `xe`, `Tmat`, `cs2` and the optical depth `kappa`.

The device holds `n_cosmologies * n_k * n_tau * len(keep)` numbers and never the full state history.

## Derivatives

Give `Omegam`, `Omegab`, `H0`, `Tcmb` or `Neff` as JAX values and the kept values become differentiable. `jax.jvp`, `jax.jacfwd` and `jax.grad` each cost one joint launch, which forms the derivatives inside the kernel by forward sensitivities:

```python
import jax.numpy as jnp

def R_today(omega_m):
    return solve_eb(planck.replace(Omegam=omega_m), k, tau, keep=("R",))[0, 0, :, -1]

dR = jax.jacfwd(R_today)(jnp.asarray(planck.Omegam))  # (64,)
```

The derivative holds the conformal age and the reionization redshift fixed. The other parameters can't be differentiated, and trying raises an error.

## Under `jit`

A cosmology is a JAX pytree whose leaves are its parameters, so `jax.jit` traces them and one compiled function serves every cosmology of the model. The wave numbers must be plain numbers, since they set the launch's shape:

```python
import jax

@jax.jit
def power_spectrum(cosmology):
    tau0 = conformal_ages(cosmology)[0]
    values = solve_eb(  # (cosmology, name, k, tau)
        cosmology,
        k,
        tau0.reshape(1),
        keep=("clxc", "clxb", "z", "vb"),
        thermo_weight=0.1,  # a looser recombination history suffices for P(k)
    )
    clxc, clxb, z, vb = jnp.unstack(values[0, :, :, -1])
    return matter_power_spectrum(
        k,
        clxc=clxc,
        clxb=clxb,
        z=z,
        vb=vb,
        tau0=tau0,
        Omegam=cosmology.Omegam,
        Omegab=cosmology.Omegab,
        A_s=cosmology.A_s,
        n_s=cosmology.n_s,
        k_p=cosmology.k_p,
    )
```

Under `jit` a model can't check its cosmologies (for instance `w_DE_0 = -1` under `W0WACDM`), since their values aren't known yet.

## Extending the model

A new species or interaction is a `Component` subclass; list it among a model's bases. It declares its parameters with an annotation, `Omega_x: ArrayLike = Parameter(0.0)` for a key and `grhox: ArrayLike = Derived(lambda grhom, Omega_x: grhom * Omega_x)` for one derived from others, so that they reach the model's constructor. A method provides the quantity it is named after; a contribution to a sum or to another component's derivative has a name of its own and says what it adds to, `@contributes("dgrho") def dgrho_x(self, ...)`, so that the model class inheriting every component has no clashing methods. To keep quantities a model does not provide, `solve_eb(..., extra=...)` appends your own components. A stateless `Component` subclass declares providers whose parameter names are the quantities they read. DISCO-CMB defines its line-of-sight sources this way, so they compile into the same kernel.
