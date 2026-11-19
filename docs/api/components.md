# Components

`discoeb.components` holds the framework that turns components into
equations, and the components of the built-in models. See [Building
models](../usage/models.md) for how to write one.

## The framework

```{eval-rst}
.. autoclass:: discoeb.components.Component
   :members: named, setting_names, provided_names

.. autoclass:: discoeb.components.Species
   :no-members:

.. autoclass:: discoeb.components.Gravity
   :no-members:

.. autoclass:: discoeb.components.Interaction
   :no-members:

.. autoclass:: discoeb.components.State
   :no-members:

.. autofunction:: discoeb.components.Parameter

.. autoclass:: discoeb.components.Setting
   :no-members:

.. autoclass:: discoeb.components.Sum
   :no-members:

.. autofunction:: discoeb.components.contributes

.. autofunction:: discoeb.components.derivative

.. autoclass:: discoeb.components.Equations
   :members: sparsity, stages, published, evaluate, initial_state, state_values, build, build_function
```

## Gravity

```{eval-rst}
.. automodule:: discoeb.components.gravity
   :no-members:

.. autoclass:: discoeb.components.gravity.Background
   :no-members:

.. autoclass:: discoeb.components.gravity.Metric
   :no-members:

.. autoclass:: discoeb.components.gravity.Curvature
   :no-members:

.. autoclass:: discoeb.components.gravity.AdiabaticMode
   :no-members:
```

## Species

```{eval-rst}
.. automodule:: discoeb.components.species
   :no-members:

.. autoclass:: discoeb.components.species.ColdDarkMatter
   :no-members:

.. autoclass:: discoeb.components.species.Baryons
   :no-members:

.. autoclass:: discoeb.components.species.Photons
   :no-members:

.. autoclass:: discoeb.components.species.MasslessNeutrinos
   :no-members:

.. autoclass:: discoeb.components.species.MassiveNeutrinos
   :no-members:

.. autoclass:: discoeb.components.species.CosmologicalConstant
   :no-members:

.. autoclass:: discoeb.components.species.DarkEnergy
   :no-members:

.. autoclass:: discoeb.components.species.Fluid
   :no-members:

.. autoclass:: discoeb.components.species.RadiationStreaming
   :no-members:

.. autoclass:: discoeb.components.species.NoRadiationStreaming
   :no-members:
```

## Interactions

```{eval-rst}
.. autoclass:: discoeb.components.interactions.ThomsonScattering
   :no-members:
```

## Recombination

```{eval-rst}
.. autoclass:: discoeb.components.recombination.Recombination
   :no-members:

.. autoclass:: discoeb.components.recombination.ReionizationDepth
   :no-members:
```
