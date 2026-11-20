# Components

```{eval-rst}
.. automodule:: discoeb.components
   :no-members:
```

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

```{toctree}
:maxdepth: 1

components/gravity
components/species
components/interactions
components/recombination
```
