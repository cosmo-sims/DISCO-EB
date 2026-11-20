# Approximations

Wherever possible, DISCO-EB avoids the use of approximations in order to
make the solver more flexible and easily generalisable to new physics.

However some approximations, such as photon and neutrino hierarchy truncation, 
are inevitable. You can also turn on the radiation streaming approximation,
which replaces the photon and massless neutrino hierarchies with a closed form solution at late times.
You can control these approximations like ordinary settings using class keywords, the `Model.with_settings(...)` helper, or environment variables.

## Hierarchy truncation

Each Boltzmann hierarchy stops at a maximum multipole and closes with the
free-streaming truncation of Ma & Bertschinger.

| Setting | Default | Hierarchy |
|---|---|---|
| `LMAX_G` | 15 | photon temperature |
| `LMAX_POL` | 15 | photon polarization |
| `LMAX_NR` | 15 | massless neutrinos |
| `LMAX_NU` | 15 | massive neutrinos, per momentum bin |
| `NQMAX` | 3 | momentum bins of the massive neutrinos |

The minimum multipole is 3. The environment variables are `DISCO_LMAX_G`,
`DISCO_NQMAX` and so on.

```python
Fine = LCDM_MNU.with_settings(LMAX_NU=20, NQMAX=5)
```

## Radiation streaming

Well inside the horizon and after decoupling, the photon and massless
neutrino hierarchies oscillate rapidly about a solution driven by the
metric. {py:class}`~discoeb.components.species.RadiationStreaming`
implements the radiation streaming approximation of CLASS ([Blas,
Lesgourgues & Tram 2011](https://arxiv.org/abs/1104.2933)). Once it
switches on, the model replaces the hierarchies by that solution:

$$
\delta = \frac{4}{k^2}\left(aH\,h' - k^2 \eta\right), \qquad \theta = -\frac{h'}{2}, \qquad \sigma = 0,
$$

and freezes their states. The switch needs $k\tau > 45$ and $\kappa'\tau <
0.2$, and stays on after $a = 0.02$.

| Setting | Default | Meaning |
|---|---|---|
| `RSA` | 1 | 0 turns the approximation off |
| `RSA_K_TAU` | 45 | the threshold in $k\tau$ |
| `RSA_KAPPA_TAU` | 0.2 | the threshold in $\kappa'\tau$ |
| `RSA_A_LATCH` | 0.02 | the scale factor after which it stays on |

```python
Exact = LCDM.with_settings(RSA=0)
Late = LCDM.with_settings(RSA_K_TAU=100.0)
```

To switch off the ability to use RSA entirely, swap in the {py:class}`~discoeb.components.species.NoRadiationStreaming` component:

```python
from discoeb.components.species import NoRadiationStreaming, RadiationStreaming

LCDM.swap(RadiationStreaming, NoRadiationStreaming, name="LCDM-noRSA")
```

## Recombination

The recombination block is RECFAST, which CLASS also offers. Its rates
evaluate their transcendental functions in single precision by default.
`DISCO_THERMO_FP32_MATH=0` switches them to double precision.

For the matter power spectrum a looser recombination history suffices.
`solve_eb(..., thermo_weight=0.1)` lowers the block's weight in the error
norm and saves steps.
