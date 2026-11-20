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
a static type checker will report the same based on type declarations.

## Setting densities

The densities evolved in the model are physical, $\omega_i = \Omega_i h^2$. To start
from $\Omega_m$, $\Omega_b$ and $H_0$ instead, use:

```python
planck = LCDM.from_density_parameters(
    Omega_m=0.3142, Omega_b=0.0494, H_0=67.32,
    A_s=2.1e-9, n_s=0.966, T_cmb=2.7255, Y_He=0.2454, N_eff=3.046,
)
```

which finds the `omega_de` that makes the model expand at `H_0` today.

## Changing parameters

A cosmology is immutable. `replace` returns a changed copy:

```python
heavier = planck.replace(omega_c=0.125)
```

`planck.to_dict()` returns the parameters as a dictionary and `LCDM.from_dict(params)` accepts a
dictionary as input.

## Reionization

`z_re` is the redshift of reionization, and defaults to 0. To compute the optical depth as an output, add
{py:class}`~discoeb.components.recombination.ReionizationDepth` to the model
and keep `kappa_reio`, whose value today is $\tau_\mathrm{reio}$:

```python
from discoeb.components.recombination import ReionizationDepth

LCDM_REIO = LCDM.add(ReionizationDepth, name="LCDM+reio")
```
