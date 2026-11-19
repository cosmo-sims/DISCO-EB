# Building models

`discoeb.models` defines the following built-in cosmological models:

| Model | Components added to `LCDM` |
|---|---|
| {py:class}`~discoeb.LCDM` | CDM, baryons, photons, massless neutrinos, a cosmological constant, Thomson scattering |
| {py:class}`~discoeb.LCDM_MNU` | {py:class}`~discoeb.components.species.MassiveNeutrinos` |
| {py:class}`~discoeb.W0WACDM` | {py:class}`~discoeb.components.species.DarkEnergy` in place of the cosmological constant |
| {py:class}`~discoeb.W0WACDM_MNU` | both |

## Constructing a new model

New cosmological models can be built by defining a new class that inherits from existing models and/or base components that define new species or physics.

```python
from discoeb import LCDM
from discoeb.components.species import MassiveNeutrinos

class LCDM_MNU(LCDM, MassiveNeutrinos):
    """LCDM with massive neutrinos."""
```

Model settings, such as the lengths of the
multipole hierarchies, are defined via class keyword arguments. The following example defines a model with fewer photon and neutrino multipoles:

```python
class Coarse(LCDM, LMAX_G=10, LMAX_NR=10):
    """LCDM with fewer photon and neutrino multipoles."""
```

You can also modify an existing model by adding, removing or swapping components, or changing its settings, as follows:

```python
from discoeb.components.species import CosmologicalConstant, DarkEnergy, Fluid

LCDM.swap(CosmologicalConstant, DarkEnergy, name="w0waCDM")
LCDM.add(Fluid.named("dr"), name="LCDM+dr")   # parameters dr_omega, dr_w, dr_cs2
LCDM.remove(CosmologicalConstant, name="CDM")
LCDM.with_settings(LMAX_G=20)
```

## Writing a new component

Components are base classes that model a particular fluid species, interaction between species, or gravitational theory. A new component is defined by creating a new class that inherits from an existing component like {py:class}`~discoeb.components.Species`. It declares its state, cosmological parameters and settings as class attributes. Its `@derivative` methods define contributions to the Einstein-Boltzmann equations, and `@contributes` methods mark summed quantities that they contribute to. 

Components consist of the following elemtents:

- `State(size, initial=...)` is a state, scalar by default. `initial`
  gives its starting value for the integration.
- `Parameter(default)` is a constructor argument of every model with the
  component. A method reads it by naming it. A method that reads only
  parameters, such as `def omega_g(self, T_cmb)`, runs once per cosmology
  before the solve, and the kernel reads its value.
- `Setting(defult)` is a compile-time constant.
- `@derivative("x")` marks a contribution to the rate of the state `x`, in
  conformal time.
- `@contributes("rho")` marks a contribution to a summed quantity. The gravity
  components read the sums `rho`, `pressure`, `delta_rho`, `momentum` and
  `stress`, and the initial conditions read `rho_matter`, `rho_radiation`
  and `rho_free_streaming`.

The methods compile to CUDA device code, so they must be written in plain Python. `math` functions and
plain loops with bounds fixed by settings are acceptable. A model missing a quantity that
one of its components' methods requires as an input will error at compile time.

For example, the following component defines a second dark matter species with a density parameter `omega_x` and an adiabatic initial condition for its density perturbation `delta_x`:

```python
from jax.typing import ArrayLike
from discoeb.components import Parameter, Species, State, contributes, derivative

class SecondDarkMatter(Species):
    delta_x = State(initial=lambda F: 0.75 * F[0])    # adiabatic mode

    omega_x: ArrayLike = Parameter(0.0)

    def rho_x(self, omega_x, a):
        return omega_x / a**3

    @contributes("rho", "rho_matter")
    def rho_second(self, rho_x):
        return rho_x

    @contributes("delta_rho")
    def delta_rho_x(self, rho_x, delta_x):
        return rho_x * delta_x

    @derivative("delta_x")
    def continuity_x(self, h_prime):
        return -0.5 * h_prime

LCDM_X = LCDM.add(SecondDarkMatter, name="LCDM+x")
```

To swap in a theory of gravity, replace the `Background` or `Metric` components with new ones:

```python
LCDM.swap(Background, MyBackground, name=...)
LCDM.swap(Metric, MyMetric, name=...).
```
