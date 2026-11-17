# Models

A model is a class whose bases are its components: the species it evolves (`discoeb.components.species`), the interactions between them (`discoeb.components.interactions`) and the metric and Friedmann sum (`discoeb.components.gravity`). An instance of a model is a cosmology. `discoeb.models` defines `LCDM` and `W0WACDM`, each with a massive-neutrino variant (`LCDM_MNU`, `W0WACDM_MNU`).

The components declare the parameters, so a model's constructor takes exactly its own, by keyword. A type checker reads them from the declarations and flags a missing or unknown one, as the constructor does at run time. `help(LCDM)` lists them. A cosmology is immutable: `planck.replace(omega_c=0.125)` gives a new one. Dictionaries go through `LCDM(**params)`, or `LCDM.from_dict(params)` for one written for another model, whose extra keys must keep their defaults.

The densities are physical ones, `omega_b`, `omega_c`, `omega_de` and `omega_k` ($\omega = \Omega h^2$), and the Hubble constant follows from them: `planck.H_0`. To start from $\Omega_m$, $\Omega_b$ and $H_0$, use `LCDM.from_density_parameters(Omega_m=..., Omega_b=..., H_0=..., ...)`, which finds the dark-energy density that makes the model expand at `H_0` today. It is plain JAX arithmetic, so a derivative in `H_0` or `Omega_m` goes through it.

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

`solve_eb` records only what you name in `keep`. A name can be a state element, such as `delta_c`, `F_2` or `psi_0_1`, which `state_names(model)` lists. It can also be any quantity the model provides, such as the comoving curvature perturbation `R`, the shear `alpha`, the expansion rate `aH` or the Thomson `opacity`. The recombination block adds the conformal time `tau`, `xe`, `Tmat`, `cs2` and the optical depth `kappa`.

The solve integrates in the scale factor, from `A_START` to 1, so you give the scale factors to record at, and `a = 1` is today for every cosmology. The conformal time is a state: keep `tau`, whose value today is the conformal age. Nothing is integrated before the solve.

The device holds `n_cosmologies * n_k * n_a * len(keep)` numbers and never the full state history.

Reionization is set by its redshift `z_re` (0, the default, for none). Its optical depth is an output: add `ReionizationDepth` (`discoeb.components.recombination`) to a model, `LCDM.add(ReionizationDepth, name="LCDM+reio")`, and keep `kappa_reio`, whose value today is $\tau_\mathrm{reio}$. The same model is the one that differentiates in `z_re`.

## Derivatives

Give `omega_b`, `omega_c`, `omega_de`, `T_cmb` or `N_eff` as JAX values and the kept values become differentiable. `jax.jvp`, `jax.jacfwd` and `jax.grad` each cost one joint launch, which forms the derivatives inside the kernel by forward sensitivities:

```python
import jax.numpy as jnp

def R_today(omega_c):
    return solve_eb(planck.replace(omega_c=omega_c), k, [1.0], keep=("R",))[0, 0, :, -1]

dR = jax.jacfwd(R_today)(jnp.asarray(planck.omega_c))  # (64,)
```

The derivative is taken at fixed scale factor, so at fixed redshift. The other parameters can't be differentiated, and trying raises an error.

## Under `jit`

A cosmology is a JAX pytree whose leaves are its parameters, so `jax.jit` traces them and one compiled function serves every cosmology of the model. The wave numbers must be plain numbers, since they set the launch's shape:

```python
import jax

@jax.jit
def power_spectrum(cosmology):
    values = solve_eb(  # (cosmology, name, k, a)
        cosmology,
        k,
        [1.0],  # today
        keep=("delta_c", "delta_b"),
        thermo_weight=0.1,  # a looser recombination history suffices for P(k)
    )
    delta_c, delta_b = jnp.unstack(values[0, :, :, -1])
    return matter_power_spectrum(
        k,
        delta_c=delta_c,
        delta_b=delta_b,
        omega_c=cosmology.omega_c,
        omega_b=cosmology.omega_b,
        A_s=cosmology.A_s,
        n_s=cosmology.n_s,
        k_p=cosmology.k_p,
    )
```

Under `jit` a model can't check its cosmologies (for instance `w_DE_0 = -1` under `W0WACDM`), since their values aren't known yet.

## Extending the model

A new species or interaction is a `Component` subclass; list it among a model's bases. It declares its parameters with an annotation, `omega_x: ArrayLike = Parameter(0.0)` for a key and `omega_g: ArrayLike = Derived(lambda T_cmb: OMEGA_G_PER_TCMB4 * T_cmb**4)` for one derived from others, so that they reach the model's constructor. A method provides the quantity it is named after. A contribution has a name of its own and says what it adds to: `@contributes("delta_rho") def delta_rho_x(self, ...)` adds to a sum, and `@derivative("delta_x") def continuity_x(self, ...)` to the rate of a state. The model class inheriting every component then has no clashing methods. To keep quantities a model does not provide, `solve_eb(..., extra=...)` appends your own components. A stateless `Component` subclass declares providers whose parameter names are the quantities they read. DISCO-CMB defines its line-of-sight sources this way, so they compile into the same kernel.
