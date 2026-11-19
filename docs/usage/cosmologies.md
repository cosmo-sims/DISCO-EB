# Instantiating cosmologies

A cosmology is an instance of a model initialised with a specific set of parameters. For example:

```python
from discoeb import LCDM

planck = LCDM(
    omega_b=0.02239, omega_c=0.1200, omega_de=0.3108,
    A_s=2.1e-9, n_s=0.966, T_cmb=2.7255, Y_He=0.2454, N_eff=3.046,
)
```

The constructor raises a `TypeError` for an unknown or missing parameter, and
a static type checker will report the same based on the declarations.

## Densities and the Hubble constant

The densities evolved in the model are physical, $\omega_i = \Omega_i h^2$. To start
from $\Omega_m$, $\Omega_b$ and $H_0$ instead, use:

```python
planck = LCDM.from_density_parameters(
    Omega_m=0.3142, Omega_b=0.0494, H_0=67.32,
    A_s=2.1e-9, n_s=0.966, T_cmb=2.7255, Y_He=0.2454, N_eff=3.046,
)
```

which finds the `omega_de` that makes the model expand at `H_0` today.

## Changing and converting

A cosmology is immutable. `replace` returns a changed copy:

```python
heavier = planck.replace(omega_c=0.125)
```

`planck.to_dict()` returns the parameters as a dictionary. `LCDM(**params)`
builds a cosmology from one, and `LCDM.from_dict(params)` accepts a
dictionary written for a larger model as long as the extra keys keep their
defaults.

## Reionization

`z_re` is the redshift of reionization, and 0, the default, means none. The
optical depth is an output. Add
{py:class}`~discoeb.components.recombination.ReionizationDepth` to a model
and keep `kappa_reio`, whose value today is $\tau_\mathrm{reio}$:

```python
from discoeb.components.recombination import ReionizationDepth

LCDM_REIO = LCDM.add(ReionizationDepth, name="LCDM+reio")
```

Take derivatives in `z_re` with this model too. Its extra state makes the
solver resolve the reionization step, and without it the derivative is
inaccurate.

## Cosmologies under `jit`

A cosmology is a JAX pytree whose leaves are its parameters. `jax.jit`
traces them, so one compiled function serves every cosmology of a model.
A traced cosmology has no values yet, so the model cannot validate it (for
instance `w_DE_0 = -1` under `W0WACDM`).
